import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from src.config import settings
from src.logger import pipeline_logger, logger


class VectorDBService:
    def __init__(self):
        # اتصال منعطف و سازگار با شبکه داکر و محلی با سقف زمان انتظار ۶۰ ثانیه
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=60,
            check_compatibility=False
        )
        self.collection_name = settings.COLLECTION_NAME
        
        # بارگذاری مدل و استخراج خودکار طول بردار با متد جدید (بدون اخطار FutureWarning)
        self.embed_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        if hasattr(self.embed_model, "get_embedding_dimension"):
            self.vector_dim = self.embed_model.get_embedding_dimension()
        else:
            self.vector_dim = self.embed_model.get_sentence_embedding_dimension()
        
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """اطمینان از وجود کالکشن و ایجاد آن بر اساس شباهت کسینوسی"""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_dim,
                        distance=Distance.COSINE
                    ),
                )
                logger.info(f"Collection '{self.collection_name}' created with dim={self.vector_dim}")
        except Exception as e:
            logger.error(f"Error ensuring collection '{self.collection_name}': {str(e)}")

    def insert_chunks(self, chunks: List[Dict[str, Any]]):
        """تبدیل چانک‌ها به امبدینگ و ثبت نقطه‌ای در Qdrant با متادیتا کامل"""
        if not chunks:
            return

        self._ensure_collection_exists()

        # استاندارد فرمت ورودی مدل‌های نوین (پیشوند passage)
        texts_to_embed = [
            f"passage: {c.get('content') or c.get('text', '')}"
            for c in chunks
        ]
        
        embeddings = self.embed_model.encode(texts_to_embed, show_progress_bar=False).tolist()

        points = []
        for chunk, vector in zip(chunks, embeddings):
            metadata = chunk.get("metadata", {})
            content = chunk.get("content") or chunk.get("text", "")
            source = metadata.get("source") or chunk.get("source", "unknown_contract.pdf")
            page = metadata.get("page") or chunk.get("page", 1)

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": content,
                        "source": source,
                        "page": page
                    }
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Successfully inserted {len(points)} chunks into '{self.collection_name}'.")

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """جستجوی شباهت معنایی با لاگینگ ساختاریافته تاخیر"""
        self._ensure_collection_exists()
        
        with pipeline_logger.track_latency("vector_search", query=query):
            query_vector = self.embed_model.encode(f"query: {query}").tolist()

            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit
            )

            results = []
            for hit in search_result.points:
                payload = hit.payload or {}
                results.append({
                    "text": payload.get("text", ""),
                    "source": payload.get("source", "unknown"),
                    "page": payload.get("page", 1),
                    "score": round(hit.score, 4) if hasattr(hit, "score") and hit.score else 0.0
                })

            return results