import os
import uuid
import hashlib
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from src.parser import extract_chunks_from_pdf
from src.logger import logger

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_contracts")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")


class VectorStoreManager:
    """مدیریت حرفه‌ای، ایمن و چندکاربره (Multi-Tenant) پایگاه داده برداری Qdrant"""

    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        # استخراج پویا و خودکار ابعاد مدل امبدینگ به جای هاردکد کردن عدد 768
        self.vector_size = self.encoder.get_sentence_embedding_dimension() or 768
        self._ensure_collection()

    def _ensure_collection(self):
        """ایجاد کلکسیون استاندارد (بدون استفاده از recreate_collection منسوخ‌شده) همراه با Payload Index"""
        try:
            if not self.client.collection_exists(COLLECTION_NAME):
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(
                    f"Collection '{COLLECTION_NAME}' created with dynamic dimension {self.vector_size}."
                )

            # ایجاد Payload Index برای سرعت بالای فیلتر و حذف بر اساس نام سند و شناسه کاربر/سازمان
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="source",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="tenant_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection or indexes: {str(e)}")

    @staticmethod
    def _generate_deterministic_id(tenant_id: str, source: str, clause_id: str, content: str) -> str:
        """تولید شناسه یکتای UUIDv5 برای جلوگیری از تداخل هم‌زمانی (Race Condition) و تکرار"""
        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()[:10]
        raw_key = f"{tenant_id}::{source}::{clause_id}::{content_hash}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_key))

    def delete_document(self, source_filename: str, tenant_id: str = "default") -> bool:
        """حذف ایمن بردارهای یک سند خاص متعلق به یک کاربر/فضای کاری مشخص"""
        try:
            self.client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source",
                                match=models.MatchValue(value=source_filename)
                            ),
                            models.FieldCondition(
                                key="tenant_id",
                                match=models.MatchValue(value=tenant_id)
                            )
                        ]
                    )
                )
            )
            logger.info(f"Deleted existing chunks for document '{source_filename}' (tenant: {tenant_id}).")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {source_filename}: {str(e)}")
            return False

    def insert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        tenant_id: str = "default",
        replace_existing_doc: bool = True,
        batch_size: int = 32
    ) -> int:
        """درج دسته‌ای (Batch)، ایمن در برابر هم‌زمانی و جلوگیری از تکرار سند در آپلود مجدد"""
        try:
            if not chunks:
                return 0

            # اگر سند قبلاً توسط همین کاربر آپلود شده، فقط چانک‌های قبلی همان سند پاک شوند (نه کل دیتابیس!)
            if replace_existing_doc:
                unique_sources = {
                    c.get("metadata", {}).get("source", "unknown.pdf") for c in chunks
                }
                for src in unique_sources:
                    self.delete_document(source_filename=src, tenant_id=tenant_id)

            total_inserted = 0
            # پردازش دسته‌ای (Batching) برای جلوگیری از کندی و قفل شدن حافظه در اسناد بزرگ
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                texts = [c["content"] for c in batch]
                embeddings = self.encoder.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False
                ).tolist()

                points = []
                for chunk, emb in zip(batch, embeddings):
                    metadata = chunk.get("metadata", {})
                    source = metadata.get("source", "unknown.pdf")
                    page = metadata.get("page", 1)
                    clause_id = metadata.get("clause_id", "general")
                    content = chunk["content"]

                    point_id = self._generate_deterministic_id(
                        tenant_id=tenant_id,
                        source=source,
                        clause_id=clause_id,
                        content=content
                    )

                    payload = {
                        "content": content,
                        "source": source,
                        "page": page,
                        "clause_id": clause_id,
                        "tenant_id": tenant_id
                    }
                    points.append(models.PointStruct(id=point_id, vector=emb, payload=payload))

                self.client.upsert(collection_name=COLLECTION_NAME, points=points)
                total_inserted += len(points)

            logger.info(f"Successfully upserted {total_inserted} chunks for tenant '{tenant_id}'.")
            return total_inserted
        except Exception as e:
            logger.error(f"Failed to insert chunks: {str(e)}")
            raise

    def list_tenant_documents(self, tenant_id: str = "default") -> List[str]:
        """دریافت لیست اسناد متعلق به یک کاربر/فضای کاری خاص جهت کنترل دسترسی"""
        try:
            records, _ = self.client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="tenant_id",
                            match=models.MatchValue(value=tenant_id)
                        )
                    ]
                ),
                limit=250,
                with_payload=["source"],
                with_vectors=False
            )
            sources = {r.payload.get("source") for r in records if r.payload and r.payload.get("source")}
            return sorted(list(sources))
        except Exception as e:
            logger.error(f"Failed to list documents for tenant '{tenant_id}': {str(e)}")
            return []

    def search(
        self,
        query: str,
        top_k: int = 3,
        tenant_id: str = "default",
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """جستجوی برداری ایزوله‌شده بر اساس سطح دسترسی کاربر (Tenant ACL)"""
        try:
            query_vector = self.encoder.encode(query, normalize_embeddings=True).tolist()
            must_conditions = [
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id)
                )
            ]
            if source_filter:
                must_conditions.append(
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source_filter)
                    )
                )

            hits = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=models.Filter(must=must_conditions),
                limit=top_k
            )

            results = []
            for hit in hits:
                payload = hit.payload or {}
                results.append({
                    "content": payload.get("content", ""),
                    "metadata": {
                        "source": payload.get("source", "unknown"),
                        "page": payload.get("page", 1),
                        "clause_id": payload.get("clause_id", "N/A"),
                        "tenant_id": payload.get("tenant_id", tenant_id)
                    },
                    "score": hit.score
                })
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []


VectorDBService = VectorStoreManager