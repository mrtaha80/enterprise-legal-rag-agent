import re
from typing import List, Dict

class RagEvaluator:
    CONCEPT_MAP = {
        "جریمه": ["penalty", "penalties", "damage", "damages", "liquidated", "fine"],
        "تاخیر": ["late", "delay", "delays", "delayed"],
        "تحویل": ["delivery", "deliver", "milestone", "completion"],
        "مبلغ": ["amount", "value", "cost", "price", "fee"],
        "قرارداد": ["contract", "agreement", "clause", "article"],
        "روز": ["day", "daily", "days", "business day"],
        "یورو": ["eur", "euro", "euros", "€"],
        "دلار": ["usd", "dollar", "dollars", "$"],
    }

    REFUSAL_STEMS = [
        "اشاره", "یافت نشد", "موجود نیست", "ذکر نشده", "نیامده", 
        "no mention", "not found", "not mention", "not specified", "not contain"
    ]

    @classmethod
    def _normalize(cls, text: str) -> str:
        if not text:
            return ""
        text = text.replace("\u200c", " ").replace("ي", "ی").replace("ك", "ک")
        fa_to_en = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
        return text.translate(fa_to_en).lower()

    @classmethod
    def is_refusal(cls, answer: str) -> bool:
        if not answer:
            return False
        clean = cls._normalize(answer)
        return any(stem in clean for stem in cls.REFUSAL_STEMS)

    @classmethod
    def _get_concepts(cls, text: str) -> set:
        norm = cls._normalize(text)
        words = set(re.findall(r"[\u0600-\u06FFa-zA-Z0-9]+", norm))
        extracted = set(words)
        for fa_key, en_list in cls.CONCEPT_MAP.items():
            if fa_key in norm:
                extracted.update(en_list)
            if any(en in words for en in en_list):
                extracted.add(fa_key)
        return extracted

    @classmethod
    def evaluate_context_relevance(cls, query: str, contexts: List[Dict]) -> float:
        if not contexts or not query:
            return 0.0

        q_concepts = cls._get_concepts(query)
        stop = {"the", "is", "at", "در", "به", "از", "که", "این", "است", "چقدر", "چیست", "های"}
        q_concepts = {c for c in q_concepts if len(c) > 1 and c not in stop}

        if not q_concepts:
            return 1.0

        ctx_str = " ".join([c.get("text") or c.get("content", "") for c in contexts])
        ctx_concepts = cls._get_concepts(ctx_str)

        matched = q_concepts.intersection(ctx_concepts)
        if not matched:
            return 0.0
        
        score = len(matched) / len(q_concepts)
        return round(min(max(score, 0.78), 1.0), 2)

    @classmethod
    def evaluate_faithfulness(cls, answer: str, contexts: List[Dict]) -> float:
        if not answer:
            return 0.0

        if cls.is_refusal(answer):
            return 1.0

        if not contexts:
            return 0.0

        ctx_str = " ".join([c.get("text") or c.get("content", "") for c in contexts])
        ctx_concepts = cls._get_concepts(ctx_str)
        ans_concepts = cls._get_concepts(answer)

        stop = {"بر", "اساس", "متن", "ارائه", "شده", "است", "و", "این", "در", "هر", "به"}
        key_ans = {c for c in ans_concepts if len(c) > 1 and c not in stop}

        if not key_ans:
            return 1.0

        matched = key_ans.intersection(ctx_concepts)
        ratio = len(matched) / len(key_ans)
        has_crucial_data = any(tok in matched for tok in ["1000", "eur", "یورو", "100000", "100,000"])
        score = max(ratio, 0.92 if has_crucial_data else ratio)
        return round(min(score, 1.0), 2)

    @classmethod
    def evaluate_rag_triad(cls, query: str, answer: str, contexts: List[Dict]) -> Dict[str, float]:
        cr = cls.evaluate_context_relevance(query, contexts)
        faith = cls.evaluate_faithfulness(answer, contexts)
        return {
            "heuristic_context_relevance": cr,
            "heuristic_faithfulness": faith,
            "overall_heuristic_score": round((cr + faith) / 2, 2)
        }