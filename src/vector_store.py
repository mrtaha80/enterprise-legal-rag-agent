import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from src.config import settings

class VectorDBService:
    def __init__(self):
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.collection_name = settings.COLLECTION_NAME
        self.embed_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                )
        except Exception as e:
            print(f"Warning checking collections: {e}")

    def insert_chunks(self, chunks: list[dict]):
        self._ensure_collection_exists()
        texts_to_embed = [f"passage: {c['content']}" for c in chunks]
        embeddings = self.embed_model.encode(texts_to_embed, show_progress_bar=True).tolist()

        points = []
        for chunk, vector in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk["content"],
                        "source": chunk["metadata"]["source"],
                        "page": chunk["metadata"]["page"]
                    }
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, limit: int = 10):
        self._ensure_collection_exists()
        
        # اگر کالکشن خالی باشد جستجو نکند
        count_info = self.client.count(collection_name=self.collection_name)
        if count_info.count == 0:
            return []

        query_vector = self.embed_model.encode(f"query: {query}").tolist()
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )
        return [
            {
                "text": hit.payload["text"],
                "source": hit.payload["source"],
                "page": hit.payload["page"],
                "score": hit.score
            }
            for hit in search_result.points
        ]