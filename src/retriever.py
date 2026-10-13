from sentence_transformers import CrossEncoder
from src.config import settings
from src.vector_store import VectorDBService

class AdvancedRetriever:
    def __init__(self, vector_db: VectorDBService):
        self.vector_db = vector_db
        self.reranker = CrossEncoder(settings.RERANKER_MODEL)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        candidates = self.vector_db.search(query=query, limit=10)
        if not candidates:
            return []

        pairs = [[query, item["text"]] for item in candidates]
        scores = self.reranker.predict(pairs)

        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)

        sorted_results = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return sorted_results[:top_k]
