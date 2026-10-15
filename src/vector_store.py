import uuid
import hashlib
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from src.config import settings
from src.logger import logger


class VectorStoreManager:
    """مدیریت یکپارچه و چندکاربره (Multi-Tenant) پایگاه داده برداری Qdrant با پشتیبانی از E5 Prefix و Pagination کامل"""

    def __init__(self):
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.encoder = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.collection_name = settings.COLLECTION_NAME
        self.use_e5_prefix = settings.USE_E5_PREFIX and ("e5" in settings.EMBEDDING_MODEL.lower())
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
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info(f"Collection '{self.collection_name}' created (dim={self.vector_size}).")

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="tenant_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection: {str(e)}")

    def _format_passage(self, text: str) -> str:
        """افزودن پیشوند استاندارد passage: برای خانواده مدل‌های E5"""
        clean = text.strip()
        if self.use_e5_prefix and not clean.lower().startswith("passage:"):
            return f"passage: {clean}"
        return clean

    def _format_query(self, query: str) -> str:
        """افزودن پیشوند استاندارد query: برای خانواده مدل‌های E5"""
        clean = query.strip()
        if self.use_e5_prefix and not clean.lower().startswith("query:"):
            return f"query: {clean}"
        return clean

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
                                match=models.MatchValue(value=source_filename),
                            ),
                            models.FieldCondition(
                                key="tenant_id",
                                match=models.MatchValue(value=tenant_id),
                            ),
                        ]
                    )
                ),
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
        batch_size: int = 32,
    ) -> int:
        """درج دسته‌ای چانک‌ها با پیشوند E5 و جایگزینی ایمن اسناد تکراری همان کاربر"""
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
                batch = chunks[i : i + batch_size]
                raw_texts = [c.get("content") or c.get("text", "") for c in batch]
                encoded_passages = [self._format_passage(t) for t in raw_texts]

                embeddings = self.encoder.encode(
                    encoded_passages,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist()

                points = []
                for chunk, content, emb in zip(batch, raw_texts, embeddings):
                    metadata = chunk.get("metadata", {})
                    source = metadata.get("source") or chunk.get("source", "unknown.pdf")
                    page = metadata.get("page") or chunk.get("page", 1)
                    clause_id = metadata.get("clause_id") or chunk.get("clause_id", "general")

                    point_id = self._generate_deterministic_id(
                        tenant_id=tenant_id,
                        source=source,
                        clause_id=clause_id,
                        content=content,
                    )

                    payload = {
                        "content": content,
                        "source": source,
                        "page": page,
                        "clause_id": clause_id,
                        "tenant_id": tenant_id,
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
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """جستجوی برداری (Dense) با پیشوند query: برای مدل E5 و فیلتر Tenant ACL"""
        try:
            formatted_query = self._format_query(query)
            query_vector = self.encoder.encode(formatted_query, normalize_embeddings=True).tolist()

            must_conditions = [
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id),
                )
            ]
            if source_filter and source_filter != "همه اسناد مجاز":
                must_conditions.append(
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source_filter),
                    )
                )

            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=models.Filter(must=must_conditions),
                limit=top_k,
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
                        "tenant_id": payload.get("tenant_id", tenant_id),
                    },
                    "score": hit.score,
                    "dense_score": hit.score,
                })
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []

    def scroll_tenant_records(
        self,
        tenant_id: str = "default",
        source_filter: Optional[str] = None,
        page_size: int = 250,
        max_records: int = 5000,
    ) -> List[Dict[str, Any]]:
        """
        واکشی کامل رکوردهای مجاز یک کاربر با پشتیبانی از Pagination واقعی (next_page_offset)
        جهت اطمینان از اینکه اسناد بزرگ بیش از ۱۰۰ چانک از جستجوی واژگانی جا نمی‌مانند.
        """
        try:
            must_conditions = [
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id),
                )
            ]
            if source_filter and source_filter != "همه اسناد مجاز":
                must_conditions.append(
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source_filter),
                    )
                )

            scroll_filter = models.Filter(must=must_conditions)
            output: List[Dict[str, Any]] = []
            next_offset = None

            while len(output) < max_records:
                batch_limit = min(page_size, max_records - len(output))
                records, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=scroll_filter,
                    limit=batch_limit,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )

                if not records:
                    break

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
                            "tenant_id": payload.get("tenant_id", tenant_id),
                        },
                    })

                if next_offset is None:
                    break

            return output
        except Exception as e:
            logger.error(f"Scroll records pagination failed: {str(e)}")
            return []

    def list_tenant_documents(self, tenant_id: str = "default") -> List[str]:
        """لیست کردن تمام نام فایل‌های آپلودشده توسط یک کاربر مشخص با پیمایش کامل صفحات"""
        records = self.scroll_tenant_records(tenant_id=tenant_id, page_size=250, max_records=5000)
        sources = {r["source"] for r in records if r.get("source")}
        return sorted(list(sources))


VectorDBService = VectorStoreManager