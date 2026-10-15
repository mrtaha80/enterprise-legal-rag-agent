import os
import uuid
import hashlib
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from src.config import settings
from src.parser import extract_chunks_from_pdf
from src.logger import logger


class VectorStoreManager:
    """مدیریت یکپارچه و چندکاربره (Multi-Tenant) پایگاه داده برداری Qdrant"""

    def __init__(self):
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.encoder = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.collection_name = settings.COLLECTION_NAME
        self.vector_size = self.encoder.get_sentence_embedding_dimension() or 768
        self._ensure_collection()

    def _ensure_collection(self):
        """ایجاد کلکسیون و ایندکس‌های Payload برای فیلتر سریع بر اساس کاربر و نام سند"""
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Collection '{self.collection_name}' created (dim={self.vector_size}).")

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="tenant_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection: {str(e)}")

    @staticmethod
    def _generate_deterministic_id(tenant_id: str, source: str, clause_id: str, content: str) -> str:
        """تولید شناسه یکتا (UUIDv5) برای جلوگیری از تداخل در محیط چندکاربره"""
        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()[:10]
        raw_key = f"{tenant_id}::{source}::{clause_id}::{content_hash}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_key))

    def delete_document(self, source_filename: str, tenant_id: str = "default") -> bool:
        """حذف ایمن بردارهای یک سند خاص در فضای کاری کاربر مشخص"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
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
            logger.info(f"Deleted document '{source_filename}' for tenant '{tenant_id}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {source_filename}: {str(e)}")
            return False

    def insert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        tenant_id: str = "default",
        clear_existing: bool = True,
        batch_size: int = 32
    ) -> int:
        """درج دسته‌ای چانک‌ها؛ در صورت فعال بودن clear_existing فقط نسخه قبلی همان سندِ همان کاربر پاک می‌شود"""
        try:
            if not chunks:
                return 0

            if clear_existing:
                unique_sources = {
                    c.get("metadata", {}).get("source") or c.get("source", "unknown.pdf")
                    for c in chunks
                }
                for src in unique_sources:
                    self.delete_document(source_filename=src, tenant_id=tenant_id)

            total_inserted = 0
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                texts = [c.get("content") or c.get("text", "") for c in batch]
                embeddings = self.encoder.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False
                ).tolist()

                points = []
                for chunk, emb in zip(batch, embeddings):
                    metadata = chunk.get("metadata", {})
                    content = chunk.get("content") or chunk.get("text", "")
                    source = metadata.get("source") or chunk.get("source", "unknown.pdf")
                    page = metadata.get("page") or chunk.get("page", 1)
                    clause_id = metadata.get("clause_id") or chunk.get("clause_id", "general")

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

                self.client.upsert(collection_name=self.collection_name, points=points)
                total_inserted += len(points)

            logger.info(f"Inserted {total_inserted} chunks into Qdrant for tenant '{tenant_id}'.")
            return total_inserted
        except Exception as e:
            logger.error(f"Failed to insert chunks: {str(e)}")
            raise

    def search(
        self,
        query: str,
        top_k: int = 3,
        tenant_id: str = "default",
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """جستجوی برداری (Dense) با اعمال فیلتر ایزوله‌سازی کاربر (Tenant ACL)"""
        try:
            query_vector = self.encoder.encode(query, normalize_embeddings=True).tolist()
            must_conditions = [
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id)
                )
            ]
            if source_filter and source_filter != "همه اسناد مجاز":
                must_conditions.append(
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source_filter)
                    )
                )

            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=models.Filter(must=must_conditions),
                limit=top_k
            )

            results = []
            for hit in hits:
                payload = hit.payload or {}
                results.append({
                    "content": payload.get("content", ""),
                    "text": payload.get("content", ""),
                    "source": payload.get("source", "unknown.pdf"),
                    "page": payload.get("page", 1),
                    "metadata": {
                        "source": payload.get("source", "unknown.pdf"),
                        "page": payload.get("page", 1),
                        "clause_id": payload.get("clause_id", "N/A"),
                        "tenant_id": payload.get("tenant_id", tenant_id)
                    },
                    "score": hit.score,
                    "dense_score": hit.score
                })
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []

    def scroll_tenant_records(
        self,
        tenant_id: str = "default",
        source_filter: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """واکشی رکوردهای مجاز یک کاربر برای بخش جستجوی واژگانی (Lexical Search)"""
        try:
            must_conditions = [
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id)
                )
            ]
            if source_filter and source_filter != "همه اسناد مجاز":
                must_conditions.append(
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source_filter)
                    )
                )

            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(must=must_conditions),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            output = []
            for r in records:
                payload = r.payload or {}
                output.append({
                    "content": payload.get("content", ""),
                    "text": payload.get("content", ""),
                    "source": payload.get("source", "unknown.pdf"),
                    "page": payload.get("page", 1),
                    "metadata": {
                        "source": payload.get("source", "unknown.pdf"),
                        "page": payload.get("page", 1),
                        "clause_id": payload.get("clause_id", "N/A"),
                        "tenant_id": payload.get("tenant_id", tenant_id)
                    }
                })
            return output
        except Exception as e:
            logger.error(f"Scroll records failed: {str(e)}")
            return []

    def list_tenant_documents(self, tenant_id: str = "default") -> List[str]:
        """لیست کردن نام فایل‌های آپلودشده توسط یک کاربر مشخص"""
        records = self.scroll_tenant_records(tenant_id=tenant_id, limit=250)
        sources = {r["source"] for r in records if r.get("source")}
        return sorted(list(sources))


# تعریف نام‌های یکسان برای سازگاری کامل با تمامی فایل‌ها و تست‌ها
VectorDBService = VectorStoreManager