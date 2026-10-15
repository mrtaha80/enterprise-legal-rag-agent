import os
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
    """مدیریت پیشرفته پایگاه داده برداری Qdrant همراه با قابلیت حذف و Re-index کردن سندها"""
    
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
            # ۱. بازسازی و پاکسازی کلکسیون
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

                texts = [c["content"] for c in chunks]
                embeddings = self.encoder.encode(texts, normalize_embeddings=True).tolist()

                points = []
                for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    point_id = total_points + idx + 1
                    payload = {
                        "content": chunk["content"],
                        "source": chunk["metadata"]["source"],
                        "page": chunk["metadata"]["page"],
                        "clause_id": chunk["metadata"]["clause_id"]
                    }
                    points.append(models.PointStruct(id=point_id, vector=emb, payload=payload))

                self.client.upsert(collection_name=COLLECTION_NAME, points=points)
                total_points += len(points)
                logger.info(f"Indexed {len(points)} chunks from document: {file_name}")

            logger.info(f"Re-indexing completed successfully. Total vectors indexed: {total_points}")
            return total_points

        except Exception as e:
            logger.error(f"Error during document re-indexing: {str(e)}")
            raise

vector_store = VectorStoreManager()
