import math
from sentence_transformers import CrossEncoder
from src.config import settings
from src.vector_store import VectorDBService
from src.logger import trace_latency, logger
from src.evaluator import RagEvaluator

class HybridRetriever:
    def __init__(self, vector_db: VectorDBService = None):
        # استفاده از همان نمونه موجود برای جلوگیری از دو بار پر شدن رم!
        self.vector_db = vector_db if vector_db is not None else VectorDBService()
        self.reranker = CrossEncoder(settings.RERANKER_MODEL)

    def _expand_bilingual_query(self, query: str) -> str:
        """غنی‌سازی کوئری فارسی با معادل‌های کلیدی انگلیسی برای اسناد بین‌المللی"""
        norm_q = RagEvaluator._normalize(query)
        en_terms = []
        for fa_word, en_list in RagEvaluator.CONCEPT_MAP.items():
            if fa_word in norm_q:
                en_terms.extend(en_list[:2])
        if en_terms:
            return f"{query} ({' '.join(en_terms)})"
        return query

    def retrieve(self, query: str, top_k: int = 3, candidate_limit: int = 8) -> list[dict]:
        expanded_query = self._expand_bilingual_query(query)

        # ۱. جستجوی برداری در Qdrant
        with trace_latency("vector_search", query=query):
            candidates = self.vector_db.search(query=expanded_query, limit=candidate_limit)

        if not candidates:
            logger.info(
                "No candidates retrieved from vector store",
                extra={"extra_fields": {"query": query, "candidates_count": 0}}
            )
            return []

        # ۲. بازچینی ترکیبی (Hybrid Score) تا بندهای انگلیسی حذف نشوند
        with trace_latency("cross_encoder_rerank", query=query) as ctx:
            pairs = [[expanded_query, c["text"]] for c in candidates]
            raw_scores = self.reranker.predict(pairs)

            for candidate, r_score in zip(candidates, raw_scores):
                # نرمال‌سازی سیگموید برای امتیاز Cross-Encoder
                norm_rerank = 1 / (1 + math.exp(-float(r_score)))
                vec_score = candidate.get("score", 0.0)
                # ترکیب ۶۰٪ شباهت معنایی E5 و ۴۰٪ امتیاز Cross-Encoder
                candidate["rerank_score"] = round((0.6 * vec_score) + (0.4 * norm_rerank), 4)

            reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
            ctx["input_candidates"] = len(candidates)
            ctx["output_chunks"] = top_k

        return reranked[:top_k]

Retriever = HybridRetriever