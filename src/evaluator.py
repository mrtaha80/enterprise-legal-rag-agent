import re
from typing import List, Dict, Any


class RagEvaluator:
    """
    ارزیاب شاخص‌های اکتشافی (Heuristic Proxy Indicator) + مجموعه بنچمارک طلایی دستی.
    توجه: شاخص‌های اکتشافی بر پایه هم‌پوشانی واژگانی/مفهومی هستند و جایگزین ارزیابی انسانی نیستند.
    """

    CONCEPT_MAP = {
        "جریمه": ["penalty", "penalties", "damage", "damages", "liquidated", "fine"],
        "تاخیر": ["late", "delay", "delays", "delayed"],
        "تحویل": ["delivery", "deliver", "milestone", "completion"],
        "مبلغ": ["amount", "value", "cost", "price", "fee"],
        "قرارداد": ["contract", "agreement", "clause", "article"],
        "فسخ": ["termination", "terminate", "cancel", "cancellation"],
        "روز": ["day", "daily", "days", "business day"],
        "یورو": ["eur", "euro", "euros", "€"],
        "دلار": ["usd", "dollar", "dollars", "$"],
    }

    REFUSAL_STEMS = [
        "اشاره", "یافت نشد", "موجود نیست", "ذکر نشده", "نیامده",
        "no mention", "not found", "not mention", "not specified", "not contain"
    ]

    # مجموعه بنچمارک دستی طلایی (Golden Benchmark Dataset) با پرسش، کانتکست، پاسخ و استناد دقیق
    GOLDEN_BENCHMARK: List[Dict[str, Any]] = [
        {
            "id": "bench_01_lease_termination",
            "query": "شرایط فسخ قرارداد اجاره تجاری به دلیل تاخیر در پرداخت اجاره‌بها چیست؟",
            "ground_truth_context": {
                "content": "ماده ۱۲-۳: در صورتی که مستأجر بیش از ۱۵ روز از تاریخ سررسید از پرداخت اجاره‌بها خودداری نماید، موجر حق دارد پس از ارسال اخطار کتبی ۱۰ روزه، نسبت به فسخ یک‌جانبه قرارداد اقدام نماید.",
                "metadata": {
                    "source": "commercial_lease_1402.pdf",
                    "page": 4,
                    "clause_id": "p4_c12_s3"
                }
            },
            "expected_answer_keywords": ["15", "روز", "اخطار", "10", "فسخ"],
            "expected_citation": {
                "source": "commercial_lease_1402.pdf",
                "page": 4,
                "clause_id": "p4_c12_s3"
            }
        },
        {
            "id": "bench_02_delay_penalty",
            "query": "مبلغ جریمه تاخیر در تحویل کالا به ازای هر روز چقدر است؟",
            "ground_truth_context": {
                "content": "بند ۸-۲: در صورت تاخیر فروشنده در تحویل تجهیزات، به ازای هر روز تاخیر مبلغ ۱۰۰۰ یورو به عنوان خسارت مقطوع از مطالبات وی کسر می‌گردد و سقف این جریمه حداکثر ۱۰ درصد مبلغ کل قرارداد است.",
                "metadata": {
                    "source": "supply_agreement_2025.pdf",
                    "page": 7,
                    "clause_id": "p7_c8_s2"
                }
            },
            "expected_answer_keywords": ["1000", "یورو", "10", "درصد"],
            "expected_citation": {
                "source": "supply_agreement_2025.pdf",
                "page": 7,
                "clause_id": "p7_c8_s2"
            }
        },
        {
            "id": "bench_03_unmentioned_clause",
            "query": "نرخ سود تسهیلات بانکی در این قرارداد چند درصد تعیین شده است؟",
            "ground_truth_context": {
                "content": "ماده ۵: طرفین متعهد به حفظ محرمانگی کلیه اسناد فنی و تجاری مبادله‌شده تا ۳ سال پس از اتمام قرارداد می‌باشند.",
                "metadata": {
                    "source": "nda_contract.pdf",
                    "page": 2,
                    "clause_id": "p2_c5_s1"
                }
            },
            "expected_answer_keywords": ["اشاره"],
            "expected_citation": {
                "source": "nda_contract.pdf",
                "page": 2,
                "clause_id": "p2_c5_s1"
            }
        }
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
        """محاسبه صادقانه نسبت مفاهیم مشترک پرسش و کانتکست (بدون کف مصنوعی 0.78)"""
        if not contexts or not query:
            return 0.0

        q_concepts = cls._get_concepts(query)
        stop = {"the", "is", "at", "در", "به", "از", "که", "این", "است", "چقدر", "چیست", "های", "را", "با"}
        q_concepts = {c for c in q_concepts if len(c) > 1 and c not in stop}

        if not q_concepts:
            return 0.0

        ctx_str = " ".join([c.get("text") or c.get("content", "") for c in contexts])
        ctx_concepts = cls._get_concepts(ctx_str)

        matched = q_concepts.intersection(ctx_concepts)
        score = len(matched) / len(q_concepts)
        return round(min(score, 1.0), 2)

    @classmethod
    def evaluate_faithfulness(cls, answer: str, contexts: List[Dict]) -> float:
        """محاسبه صادقانه هم‌پوشانی واژگان کلیدی پاسخ با کانتکست (بدون پرش مصنوعی روی اعداد)"""
        if not answer:
            return 0.0

        if cls.is_refusal(answer):
            return 1.0

        if not contexts:
            return 0.0

        ctx_str = " ".join([c.get("text") or c.get("content", "") for c in contexts])
        ctx_concepts = cls._get_concepts(ctx_str)
        ans_concepts = cls._get_concepts(answer)

        stop = {"بر", "اساس", "متن", "ارائه", "شده", "است", "و", "این", "در", "هر", "به", "از", "که", "می", "شود"}
        key_ans = {c for c in ans_concepts if len(c) > 1 and c not in stop}

        if not key_ans:
            return 0.0

        matched = key_ans.intersection(ctx_concepts)
        ratio = len(matched) / len(key_ans)
        return round(min(ratio, 1.0), 2)

    @classmethod
    def evaluate_rag_triad(cls, query: str, answer: str, contexts: List[Dict]) -> Dict[str, Any]:
        cr = cls.evaluate_context_relevance(query, contexts)
        faith = cls.evaluate_faithfulness(answer, contexts)
        overall = round((cr + faith) / 2, 2)
        return {
            # کلیدهای صادقانه و استاندارد برای نمایش در UI و مستندات
            "heuristic_context_relevance_indicator": cr,
            "heuristic_faithfulness_indicator": faith,
            "heuristic_overall_indicator": overall,
            "is_heuristic_proxy": True,
            # حفظ کلیدهای قدیمی برای سازگاری با تست‌های موجود در Pytest
            "context_relevance": cr,
            "faithfulness": faith,
            "overall_rag_score": overall
        }

    @classmethod
    def evaluate_against_golden_Case(
        cls,
        case: Dict[str, Any],
        generated_answer: str,
        retrieved_contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """بررسی دقیق یک نمونه از بنچمارک طلایی شامل صحت کلمات کلیدی پاسخ و دقت Citation"""
        norm_ans = cls._normalize(generated_answer)
        expected_kw = [cls._normalize(k) for k in case.get("expected_answer_keywords", [])]
        matched_kw = [k for k in expected_kw if k in norm_ans]
        keyword_recall = round(len(matched_kw) / max(len(expected_kw), 1), 2)

        exp_cit = case.get("expected_citation", {})
        citation_matched = False
        for ctx in retrieved_contexts:
            meta = ctx.get("metadata", {})
            src = meta.get("source") or ctx.get("source")
            page = meta.get("page") or ctx.get("page")
            clause_id = meta.get("clause_id") or ctx.get("clause_id")
            if (
                src == exp_cit.get("source")
                and page == exp_cit.get("page")
                and clause_id == exp_cit.get("clause_id")
            ):
                citation_matched = True
                break

        return {
            "case_id": case.get("id"),
            "keyword_recall": keyword_recall,
            "citation_accuracy": 1.0 if citation_matched else 0.0,
            "passed": (keyword_recall >= 0.75) and citation_matched
        }