import re
from typing import List, Dict, Any, Optional, Set
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from src.config import settings
from src.vector_store import VectorDBService
from src.logger import logger


class HybridRetriever:
    """
    موتور بازیابی ترکیبی (Dense Vector Search + Normalized Lexical Overlap + RRF Fusion + Cross-Encoder Reranking)
    با پشتیبانی از صفحه‌بندی کامل (Full Pagination) و نرمال‌سازی نویسه‌های فارسی.
    """

    STOPWORDS: Set[str] = {
        "در", "به", "از", "که", "این", "ان", "آن", "با", "را", "برای", "تا", "بر",
        "است", "هست", "بود", "شد", "شود", "می", "های", "ها", "یا", "و", "هر",
        "چقدر", "چیست", "چند", "کدام", "چه", "آیا", "the", "is", "in", "at", "of",
        "on", "and", "or", "to", "for", "with", "by", "an", "a", "what", "how",
    }

    def __init__(
        self,
        vector_db: Optional[VectorDBService] = None,
        qdrant_client: Optional[QdrantClient] = None,
        encoder: Optional[SentenceTransformer] = None,
    ):
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

    @classmethod
    def _normalize_and_tokenize(cls, text: str) -> List[str]:
        """
        نرمال‌سازی متن فارسی و انگلیسی:
        - یکسان‌سازی «ی/ي» و «ک/ك»
        - تبدیل نیم‌فاصله به فاصله استاندارد
        - تبدیل ارقام فارسی و عربی به انگلیسی
        - حذف علائم نگارشی و ایست‌واژه‌ها
        """
        if not text:
            return []

        cleaned = text.replace("\u200c", " ").replace("ي", "ی").replace("ك", "ک")
        digit_map = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
        cleaned = cleaned.translate(digit_map).lower()

        raw_tokens = re.findall(r"[\u0600-\u06FFa-z0-9]+", cleaned)
        return [tok for tok in raw_tokens if len(tok) > 1 and tok not in cls.STOPWORDS]

    @classmethod
    def _lexical_score(cls, query: str, text: str) -> float:
        """محاسبه امتیاز هم‌پوشانی واژگانی نرمال‌شده همراه با پاداش تطابق ارقام و مواد قانونی"""
        q_tokens = cls._normalize_and_tokenize(query)
        t_tokens = set(cls._normalize_and_tokenize(text))

        if not q_tokens or not t_tokens:
            return 0.0

        q_set = set(q_tokens)
        matched = q_set.intersection(t_tokens)
        if not matched:
            return 0.0

        base_overlap = len(matched) / len(q_set)
        # وزن‌دهی بیشتر به تطابق دقیق اعداد (شماره ماده، بند، مبالغ یا روزها)
        numeric_matches = sum(1 for tok in matched if tok.isdigit())
        bonus = min(0.15, numeric_matches * 0.05)

        return min(base_overlap + bonus, 1.0)

    @staticmethod
    def _reciprocal_rank_fusion(
        dense_results: List[Dict[str, Any]],
        lexical_results: List[Dict[str, Any]],
        k: int = 60,
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
            d = dict(doc_map[doc_key])
            d["rrf_score"] = round(score, 6)
            fused_results.append(d)

        return fused_results

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        candidate_pool: int = 10,
        tenant_id: str = "default",
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """بازیابی ترکیبی با پیمایش صفحه‌بندی‌شده کل چانک‌های Tenant و ادغام با RRF"""
        try:
            # ۱. جستجوی معنایی (Dense Search)
            dense_results = self.vector_db.search(
                query=query,
                top_k=candidate_pool,
                tenant_id=tenant_id,
                source_filter=source_filter,
            )

            # ۲. جستجوی واژگانی نرمال‌شده روی کل رکوردهای مجاز کاربر (با Pagination کامل)
            all_records = self.vector_db.scroll_tenant_records(
                tenant_id=tenant_id,
                source_filter=source_filter,
                page_size=250,
                max_records=5000,
            )

            lexical_candidates = []
            for record in all_records:
                content = record.get("content", "")
                score = self._lexical_score(query, content)
                if score > 0.0:
                    item = dict(record)
                    item["lexical_score"] = round(score, 4)
                    lexical_candidates.append(item)

            lexical_candidates = sorted(
                lexical_candidates, key=lambda x: x["lexical_score"], reverse=True
            )[:candidate_pool]

            if not lexical_candidates:
                lexical_candidates = dense_results

            # ۳. ادغام با الگوریتم RRF
            fused_candidates = self._reciprocal_rank_fusion(dense_results, lexical_candidates, k=60)
            if not fused_candidates:
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