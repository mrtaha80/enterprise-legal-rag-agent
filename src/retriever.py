import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.logger import logger

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_contracts")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")


class HybridRetriever:
    """موتور بازیابی ترکیبی واقعی (Dense + Lexical + RRF + Cross-Encoder) با کنترل دسترسی چندکاربره"""

    def __init__(self, qdrant_client: QdrantClient, encoder: SentenceTransformer):
        self.client = qdrant_client
        self.encoder = encoder
        logger.info(f"Loading Cross-Encoder model: {RERANKER_MODEL_NAME}")
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        """محاسبه امتیاز واژگانی برای تطابق دقیق بندها، مواد و اعداد"""
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
        """ادغام نتایج Dense و Lexical با الگوریتم استاندارد RRF بر مبنای کلید یکتا"""
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
        """اجرای جریان کامل Hybrid Retrieval با اعمال فیلتر امنیتی Tenant و سند"""
        try:
            must_conditions = [
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id)
                )
            ]
            if source_filter and source_filter != "همه اسناد مجاز":
                must_conditions.append(
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source_filter)
                    )
                )

            qdrant_filter = models.Filter(must=must_conditions)

            # ۱. جستجوی معنایی (Dense Search) با فیلتر کنترل دسترسی
            query_vector = self.encoder.encode(query, normalize_embeddings=True).tolist()
            dense_hits = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=candidate_pool
            )

            dense_results = []
            for hit in dense_hits:
                payload = hit.payload or {}
                dense_results.append({
                    "content": payload.get("content", ""),
                    "metadata": {
                        "source": payload.get("source", "unknown"),
                        "page": payload.get("page", 1),
                        "clause_id": payload.get("clause_id", "N/A")
                    },
                    "dense_score": hit.score
                })

            # ۲. جستجوی واژگانی (Lexical Search) محدود به اسناد همان Tenant
            all_records, _ = self.client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=qdrant_filter,
                limit=100,
                with_payload=True,
                with_vectors=False
            )

            lexical_candidates = []
            for record in all_records:
                payload = record.payload or {}
                content = payload.get("content", "")
                score = self._lexical_score(query, content)
                if score > 0.0:
                    lexical_candidates.append({
                        "content": content,
                        "metadata": {
                            "source": payload.get("source", "unknown"),
                            "page": payload.get("page", 1),
                            "clause_id": payload.get("clause_id", "N/A")
                        },
                        "lexical_score": score
                    })

            lexical_candidates = sorted(
                lexical_candidates, key=lambda x: x["lexical_score"], reverse=True
            )[:candidate_pool]

            if not lexical_candidates:
                lexical_candidates = dense_results

            # ۳. ادغام با الگوریتم RRF
            fused_candidates = self._reciprocal_rank_fusion(dense_results, lexical_candidates, k=60)
            if not fused_candidates:
                logger.warning(f"No candidates found in hybrid retrieval for tenant '{tenant_id}'.")
                return []

            # ۴. رتبه‌بندی مجدد با Cross-Encoder
            pairs = [[query, item["content"]] for item in fused_candidates]
            rerank_scores = self.reranker.predict(pairs)

            for idx, item in enumerate(fused_candidates):
                item["rerank_score"] = float(rerank_scores[idx])

            ranked_results = sorted(fused_candidates, key=lambda x: x["rerank_score"], reverse=True)
            return ranked_results[:top_k]

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {str(e)}")
            return []