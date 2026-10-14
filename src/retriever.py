from sentence_transformers import CrossEncoder
from src.config import settings
from src.vector_store import VectorDBService

class HybridRetriever:
    def __init__(self):
        self.vector_db = VectorDBService()
        self.reranker = CrossEncoder(settings.RERANKER_MODEL)

    def retrieve(self, query: str, top_k: int = 3, candidate_limit: int = 10) -> list[dict]:
        candidates = self.vector_db.search(query=query, limit=candidate_limit)
        if not candidates:
            return []

        # آماده‌سازی جفت‌ها برای Cross-Encoder Reranker
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        # مرتب‌سازی نزولی بر اساس امتیاز بازچینی
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

# برای سازگاری با کدهای قبلی در صورت ایمپورت Retriever
Retriever = HybridRetriever