import os
from typing import List, Dict, Any
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
    """مدیریت پایگاه داده برداری Qdrant با پشتیبانی از نام‌گذاری‌های سازگار با بخش‌های مختلف"""
    
    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self._ensure_collection()

    def _ensure_collection(self):
        """بررسی و ایجاد کلکسیون در صورت عدم وجود"""
        try:
            if not self.client.collection_exists(COLLECTION_NAME):
                self.client.recreate_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=768,  # ابعاد مدل multilingual-e5-base
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Collection '{COLLECTION_NAME}' created successfully in Qdrant.")
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection: {str(e)}")

    def insert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """درج چانک‌ها همراه با امبدینگ در Qdrant (سازگار با app.py)"""
        try:
            if not chunks:
                return 0
            
            texts = [c["content"] for c in chunks]
            embeddings = self.encoder.encode(texts, normalize_embeddings=True).tolist()

            # گرفتن تعداد پوینت‌های موجود برای تعیین ID شروع
            existing_count = self.client.count(collection_name=COLLECTION_NAME).count

            points = []
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                point_id = existing_count + idx + 1
                metadata = chunk.get("metadata", {})
                payload = {
                    "content": chunk["content"],
                    "source": metadata.get("source", "unknown.pdf"),
                    "page": metadata.get("page", 1),
                    "clause_id": metadata.get("clause_id", "general")
                }
                points.append(models.PointStruct(id=point_id, vector=emb, payload=payload))

            self.client.upsert(collection_name=COLLECTION_NAME, points=points)
            logger.info(f"Successfully inserted {len(points)} chunks into Qdrant.")
            return len(points)
        except Exception as e:
            logger.error(f"Failed to insert chunks: {str(e)}")
            raise

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """جستجوی برداری دِنس (سازگار با retriever.py)"""
        try:
            query_vector = self.encoder.encode(query, normalize_embeddings=True).tolist()
            hits = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
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
                        "clause_id": payload.get("clause_id", "N/A")
                    },
                    "score": hit.score
                })
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []

    def delete_document(self, source_filename: str) -> bool:
        """حذف کامل تمام بردارها و چانک‌های یک سند خاص بر اساس نام فایل"""
        try:
            self.client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source",
                                match=models.MatchValue(value=source_filename)
                            )
                        ]
                    )
                )
            )
            logger.info(f"Successfully deleted all chunks for document: {source_filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {source_filename}: {str(e)}")
            return False

    def reindex_all_documents(self, data_dir: str = "/app/data") -> int:
        """پاکسازی کامل و ایندکس مجدد تمام فایل‌های PDF موجود در پوشه داده‌ها"""
        try:
            if self.client.collection_exists(COLLECTION_NAME):
                self.client.delete_collection(COLLECTION_NAME)
            
            self._ensure_collection()
            
            if not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)
                logger.warning(f"Data directory '{data_dir}' was missing and has been created.")
                return 0

            pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
            if not pdf_files:
                logger.info("No PDF files found in data directory for re-indexing.")
                return 0

            total_points = 0
            for file_name in pdf_files:
                file_path = os.path.join(data_dir, file_name)
                chunks = extract_chunks_from_pdf(file_path, source_name=file_name)
                if not chunks:
                    continue
                total_points += self.insert_chunks(chunks)

            logger.info(f"Re-indexing completed successfully. Total vectors indexed: {total_points}")
            return total_points

        except Exception as e:
            logger.error(f"Error during document re-indexing: {str(e)}")
            raise

# تعریف آلیاس برای سازگاری کامل با هر نام‌گذاری احتمالی در app.py
VectorDBService = VectorStoreManager
vector_store = VectorStoreManager()
