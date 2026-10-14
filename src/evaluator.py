import re
from typing import List, Dict

class RagEvaluator:
    @staticmethod
    def _tokenize(text: str) -> set:
        """پشتیبانی هم‌زمان از کلمات فارسی، لاتین و ارقام عددی"""
        tokens = re.findall(r"[\u0600-\u06FF\w\d]+", text.lower())
        stop_words = {
            "the", "is", "at", "which", "on", "a", "an", "and", "or", "in", "to", "for", "of",
            "در", "به", "از", "که", "این", "است", "با", "برای", "شد", "شده", "را", "یا", "تا", "بر", "هر", "یک"
        }
        return {t for t in tokens if len(t) > 1 and t not in stop_words}

    @classmethod
    def evaluate_context_relevance(cls, query: str, contexts: List[Dict]) -> float:
        if not contexts or not query:
            return 0.0

        query_tokens = cls._tokenize(query)
        if not query_tokens:
            return 1.0

        combined = " ".join([c["text"] for c in contexts])
        context_tokens = cls._tokenize(combined)

        overlap = query_tokens.intersection(context_tokens)
        score = len(overlap) / len(query_tokens)
        return min(round(score, 2), 1.0)

    @classmethod
    def evaluate_faithfulness(cls, answer: str, contexts: List[Dict]) -> float:
        if not answer:
            return 0.0

        refusal_markers = ["یافت نشد", "اشاره‌ای نشده", "no mention", "not found"]
        if any(marker in answer.lower() for marker in refusal_markers):
            return 1.0

        if not contexts:
            return 0.0

        answer_tokens = cls._tokenize(answer)
        if not answer_tokens:
            return 1.0

        combined = " ".join([c["text"] for c in contexts])
        context_tokens = cls._tokenize(combined)

        supported = answer_tokens.intersection(context_tokens)
        score = len(supported) / len(answer_tokens)
        return min(round(score, 2), 1.0)

    @classmethod
    def evaluate_rag_triad(cls, query: str, answer: str, contexts: List[Dict]) -> Dict[str, float]:
        cr_score = cls.evaluate_context_relevance(query, contexts)
        faith_score = cls.evaluate_faithfulness(answer, contexts)
        overall = round((cr_score + faith_score) / 2, 2)
        return {
            "context_relevance": cr_score,
            "faithfulness": faith_score,
            "overall_rag_score": overall
        }