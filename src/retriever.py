from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from src.config import settings
from src.vector_store import VectorDBService
from src.logger import logger


class HybridRetriever:
    """موتور بازیابی ترکیبی واقعی (Dense + Lexical + RRF + Cross-Encoder) سازگار با VectorDBService"""

    def __init__(
        self,
        vector_db: Optional[VectorDBService] = None,
        qdrant_client: Optional[QdrantClient] = None,
        encoder: Optional[SentenceTransformer] = None
    ):
        # پشتیبانی هم‌زمان از هر دو امضای ورودی (جهت جلوگیری از خطا در تست‌ها و app.py)
        if vector_db is not None:
            self.vector_db = vector_db
        else:
            self.vector_db = VectorDBService()
            if qdrant_client is not None:
                self.vector_db.client = qdrant_client
            if encoder is not None:
                self.vector_db.encoder = encoder

        logger.info(f"Loading Cross-Encoder model: {settings.RERANKER_MODEL}")
        self.reranker = CrossEncoder(settings.RERANKER_MODEL)

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        query_tokens = set(query.lower().split())
        text_tokens = set(text.lower().split())
        if not query_tokens:
            return 0.0
        intersection = query_tokens.intersection(text_tokens)
        return len(intersection) / len(query_tokens)

    @staticmethod
    def _reciprocal_rank_fusion(
        dense_results: List[Dict[str, Any]],
        lexical_results: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        rrf_scores = {}
        doc_map = {}

        for rank, doc in enumerate(dense_results, start=1):
            meta = doc.get("metadata", {})
            doc_key = f"{meta.get('source')}::{meta.get('clause_id')}::{doc['content'][:50]}"
            doc_map[doc_key] = doc
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + (1.0 / (k + rank))

        for rank, doc in enumerate(lexical_results, start=1):
            meta = doc.get("metadata", {})
            doc_key = f"{meta.get('source')}::{meta.get('clause_id')}::{doc['content'][:50]}"
            doc_map[doc_key] = doc
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + (1.0 / (k + rank))

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        fused_results = []
        for doc_key, score in sorted_docs:
            d = doc_map[doc_key]
            d["rrf_score"] = score
            fused_results.append(d)

        return fused_results

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        candidate_pool: int = 10,
        tenant_id: str = "default",
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """بازیابی ترکیبی با رعایت کامل سطح دسترسی کاربر (tenant_id)"""
        try:
            # ۱. جستجوی معنایی از طریق متد استاندارد vector_db.search
            dense_results = self.vector_db.search(
                query=query,
                top_k=candidate_pool,
                tenant_id=tenant_id,
                source_filter=source_filter
            )

            # ۲. جستجوی واژگانی (Lexical) روی اسناد مجاز همان کاربر
            all_records = self.vector_db.scroll_tenant_records(
                tenant_id=tenant_id,
                source_filter=source_filter,
                limit=100
            )

            lexical_candidates = []
            for record in all_records:
                content = record.get("content", "")
                score = self._lexical_score(query, content)
                if score > 0.0:
                    item = dict(record)
                    item["lexical_score"] = score
                    lexical_candidates.append(item)

            lexical_candidates = sorted(
                lexical_candidates, key=lambda x: x["lexical_score"], reverse=True
            )[:candidate_pool]

            if not lexical_candidates:
                lexical_candidates = dense_results

            # ۳. ترکیب نتایج با الگوریتم RRF
            fused_candidates = self._reciprocal_rank_fusion(dense_results, lexical_candidates, k=60)
            if not fused_candidates:
                return []

            # ۴. بازچینی با Cross-Encoder
            pairs = [[query, item["content"]] for item in fused_candidates]
            rerank_scores = self.reranker.predict(pairs)

            for idx, item in enumerate(fused_candidates):
                item["rerank_score"] = float(rerank_scores[idx])

            ranked_results = sorted(fused_candidates, key=lambda x: x["rerank_score"], reverse=True)
            return ranked_results[:top_k]

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {str(e)}")
            return []