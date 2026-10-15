import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from src.logger import logger

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_contracts")

DENSE_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-h6-ms-marco")

class HybridRetriever:
    """موتور بازیابی ترکیبی واقعی (True Hybrid Retrieval) شامل Dense + Lexical Sparse + RRF + Cross-Encoder Reranking"""
    
    def __init__(self, qdrant_client: QdrantClient, encoder: SentenceTransformer):
        self.client = qdrant_client
        self.encoder = encoder
        logger.info(f"Loading Cross-Encoder model: {RERANKER_MODEL_NAME}")
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)

    def _lexical_score(self, query: str, text: str) -> float:
        """محاسبه امتیاز واژگانی (ساده‌شده‌ی مبتنی بر هم‌پوشانی کلمات کلیدی برای تطابق دقیق بندها و اعداد)"""
        query_tokens = set(query.lower().split())
        text_tokens = set(text.lower().split())
        if not query_tokens:
            return 0.0
        intersection = query_tokens.intersection(text_tokens)
        return len(intersection) / len(query_tokens)

    def _reciprocal_rank_fusion(self, dense_results: List[Dict[str, Any]], lexical_results: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
        """ادغام نتایج Dense و Lexical با استفاده از الگوریتم استاندارد RRF"""
        rrf_scores = {}
        doc_map = {}

        # امتیازدهی RRF برای نتایج Dense
        for rank, doc in enumerate(dense_results, start=1):
            doc_id = doc["content"]
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        # امتیازدهی RRF برای نتایج Lexical
        for rank, doc in enumerate(lexical_results, start=1):
            doc_id = doc["content"]
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        # مرتب‌سازی بر اساس مجموع امتیازات RRF
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for doc_id, score in sorted_docs:
            d = doc_map[doc_id]
            d["rrf_score"] = score
            fused_results.append(d)

        return fused_results

    def retrieve(self, query: str, top_k: int = 3, candidate_pool: int = 10) -> List[Dict[str, Any]]:
        """اجرای جریان کامل Hybrid Retrieval همراه با RRF و Cross-Encoder Reranking"""
        try:
            # ۱. جستجوی معنایی (Dense Search) در Qdrant
            query_vector = self.encoder.encode(query, normalize_embeddings=True).tolist()
            dense_hits = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
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

            # ۲. جستجوی واژگانی (Lexical/Sparse Search) از بین همان کاندیداها یا کلکسیون
            # برای پایداری، از بین رکوردهای موجود در دیتابیس اسکور واژگانی می‌گیریم
            all_records, _ = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=50,
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

            # مرتب‌سازی کاندیداهای واژگانی بر اساس امتیاز کلیدواژه
            lexical_candidates = sorted(lexical_candidates, key=lambda x: x["lexical_score"], reverse=True)[:candidate_pool]

            # اگر جستجوی واژگانی نتیجه نداشت، از همان دنس استفاده می‌کنیم تا پایپلاین متوقف نشود
            if not lexical_candidates:
                lexical_candidates = dense_results

            # ۳. ترکیب نتایج با الگوریتم RRF (True Hybrid Fusion)
            fused_candidates = self._reciprocal_rank_fusion(dense_results, lexical_candidates, k=60)

            if not fused_candidates:
                logger.warning("No candidates found in hybrid retrieval pool.")
                return []

            # ۴. مرحله نهایی ریرنک با Cross-Encoder عمیق معنایی
            pairs = [[query, item["content"]] for item in fused_candidates]
            rerank_scores = self.reranker.predict(pairs)

            for idx, item in enumerate(fused_candidates):
                item["rerank_score"] = float(rerank_scores[idx])

            # مرتب‌سازی نهایی بر اساس امتیاز Cross-Encoder
            ranked_results = sorted(fused_candidates, key=lambda x: x["rerank_score"], reverse=True)

            logger.info(f"Hybrid retrieval executed successfully. Retrieved {len(ranked_results[:top_k])} top documents.")
            return ranked_results[:top_k]

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {str(e)}")
            return []
