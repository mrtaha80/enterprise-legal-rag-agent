from sentence_transformers import CrossEncoder
from src.config import settings
from src.vector_store import VectorDBService
from src.logger import trace_latency, logger

class HybridRetriever:
    def __init__(self):
        self.vector_db = VectorDBService()
        self.reranker = CrossEncoder(settings.RERANKER_MODEL)

    def retrieve(self, query: str, top_k: int = 3, candidate_limit: int = 10) -> list[dict]:
        # ۱. ردگیری زمان جستجوی برداری در پایگاه داده Qdrant
        with trace_latency("vector_search", query=query):
            candidates = self.vector_db.search(query=query, limit=candidate_limit)

        if not candidates:
            logger.info(
                "No candidates retrieved from vector store",
                extra={"extra_fields": {"query": query, "candidates_count": 0}}
            )
            return []

        # ۲. ردگیری زمان بازچینی (Re-ranking) توسط Cross-Encoder
        with trace_latency("cross_encoder_rerank", query=query) as ctx:
            pairs = [[query, c["text"]] for c in candidates]
            scores = self.reranker.predict(pairs)

            for candidate, score in zip(candidates, scores):
                candidate["rerank_score"] = float(score)

            reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
            ctx["input_candidates"] = len(candidates)
            ctx["output_chunks"] = top_k

        return reranked[:top_k]

# آلیاس جهت حفظ سازگاری
Retriever = HybridRetriever