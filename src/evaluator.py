import re
from typing import List, Dict

class RagEvaluator:
    @staticmethod
    def _tokenize(text: str) -> set:
        """استخراج کلمات معنادار به صورت ساده و مستقل از زبان"""
        words = re.findall(r"\w+", text.lower())
        # حذف واژه‌های ایست عمومی ساده
        stop_words = {"the", "is", "at", "which", "on", "a", "an", "and", "or", "in", "to", "for", "of", "در", "به", "از", "که", "این", "است"}
        return {w for w in words if len(w) > 1 and w not in stop_words}

    @classmethod
    def evaluate_context_relevance(cls, query: str, contexts: List[Dict]) -> float:
        """ارزیابی میزان پوشش کلمات کلیدی پرسش در متن‌های بازیابی‌شده"""
        if not contexts or not query:
            return 0.0

        query_tokens = cls._tokenize(query)
        if not query_tokens:
            return 1.0

        combined_context = " ".join([c["text"] for c in contexts])
        context_tokens = cls._tokenize(combined_context)

        overlap = query_tokens.intersection(context_tokens)
        score = len(overlap) / len(query_tokens)
        return round(score, 3)

    @classmethod
    def evaluate_faithfulness(cls, answer: str, contexts: List[Dict]) -> float:
        """بررسی وفاداری پاسخ نسبت به مستندات و ردگیری توهم (Grounding)"""
        if not answer:
            return 0.0
        
        # اگر پاسخ امتناع از پاسخ‌گویی باشد (عدم توهم)
        refusal_markers = ["یافت نشد", "اشاره‌ای نشده", "no mention", "not found"]
        if any(marker in answer.lower() for marker in refusal_markers):
            return 1.0

        if not contexts:
            return 0.0

        answer_tokens = cls._tokenize(answer)
        if not answer_tokens:
            return 1.0

        combined_context = " ".join([c["text"] for c in contexts])
        context_tokens = cls._tokenize(combined_context)

        # درصد کلمات کلیدی پاسخ که در کانتکست وجود دارند
        supported_tokens = answer_tokens.intersection(context_tokens)
        score = len(supported_tokens) / len(answer_tokens)
        return round(score, 3)

    @classmethod
    def evaluate_rag_triad(cls, query: str, answer: str, contexts: List[Dict]) -> Dict[str, float]:
        """ارزیابی جامع و ارائه کارنامه کیفیتی پایپ‌لاین"""
        cr_score = cls.evaluate_context_relevance(query, contexts)
        faith_score = cls.evaluate_faithfulness(answer, contexts)
        overall = round((cr_score + faith_score) / 2, 3)

        return {
            "context_relevance": cr_score,
            "faithfulness": faith_score,
            "overall_rag_score": overall
        }
        
   