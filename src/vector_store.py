import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from src.config import settings
from src.logger import pipeline_logger, logger


class VectorDBService:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=60,
            check_compatibility=False
        )
        self.collection_name = settings.COLLECTION_NAME
        
        self.embed_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        if hasattr(self.embed_model, "get_embedding_dimension"):
            self.vector_dim = self.embed_model.get_embedding_dimension()
        else:
            self.vector_dim = self.embed_model.get_sentence_embedding_dimension()
        
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
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

    def reset_collection(self):
        """پاکسازی کالکشن برای جلوگیری از انباشت چانک‌های تکراری هنگام آپلود مجدد"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except Exception:
            pass
        self._ensure_collection_exists()

    def insert_chunks(self, chunks: List[Dict[str, Any]], clear_existing: bool = True):
        if not chunks:
            return

        if clear_existing:
            self.reset_collection()
        else:
            self._ensure_collection_exists()

        texts_to_embed = [
            f"passage: {c.get('content') or c.get('text', '')}"
            for c in chunks
        ]
        
        embeddings = self.embed_model.encode(texts_to_embed, show_progress_bar=False).tolist()

        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            metadata = chunk.get("metadata", {})
            content = (chunk.get("content") or chunk.get("text", "")).strip()
            source = metadata.get("source") or chunk.get("source", "unknown_contract.pdf")
            page = metadata.get("page") or chunk.get("page", 1)
            clause_id = metadata.get("clause_id", f"p{page}_c{idx+1}")

            # تولید ID یکتا بر اساس محتوا تا هیچ متنی دوبار ذخیره نشود
            deterministic_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source}_{page}_{content[:100]}"))

            points.append(
                PointStruct(
                    id=deterministic_id,
                    vector=vector,
                    payload={
                        "text": content,
                        "source": source,
                        "page": page,
                        "clause_id": clause_id
                    }
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Successfully inserted {len(points)} unique chunks into '{self.collection_name}'.")

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        self._ensure_collection_exists()
        
        with pipeline_logger.track_latency("vector_search", query=query):
            query_vector = self.embed_model.encode(f"query: {query}").tolist()

            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit
            )

            results = []
            seen_texts = set()
            for hit in search_result.points:
                payload = hit.payload or {}
                text = payload.get("text", "").strip()
                # جلوگیری از برگشت چانک‌های تکراری در نتایج
                if not text or text in seen_texts:
                    continue
                seen_texts.add(text)

                results.append({
                    "text": text,
                    "source": payload.get("source", "unknown"),
                    "page": payload.get("page", 1),
                    "metadata": {"clause_id": payload.get("clause_id", "ثبت‌نشده")},
                    "score": round(hit.score, 4) if hasattr(hit, "score") and hit.score else 0.0
                })

            return results