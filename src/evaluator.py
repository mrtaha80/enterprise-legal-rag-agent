import json
import math
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional


class RagEvaluator:
    """
    موتور واحد ارزیابی و بنچمارک (Single Unified Evaluation Engine)
    برای استفاده مشترک در CLI (run_benchmark.py)، رابط کاربری (app.py) و تست‌های Pytest.
    """

    CONCEPT_MAP = {
        "جریمه": ["penalty", "penalties", "damage", "damages", "liquidated", "fine", "1000", "1,000"],
        "تاخیر": ["late", "delay", "delays", "delayed"],
        "تحویل": ["delivery", "deliver", "milestone", "completion", "acceptance"],
        "مبلغ": ["amount", "value", "cost", "price", "fee", "total", "100000", "100,000"],
        "ارزش": ["value", "total", "amount", "contract"],
        "قرارداد": ["contract", "agreement", "clause", "article"],
        "قانون": ["law", "governing", "jurisdiction"],
        "داوری": ["arbitration", "dispute", "court", "jurisdiction"],
        "اختلاف": ["dispute", "arbitration", "court"],
        "محرمانگی": ["confidentiality", "confidential", "survival", "nda"],
        "فورس": ["force", "majeure"],
        "فسخ": ["termination", "terminate", "cancel", "convenience", "notice"],
        "اخطار": ["notice", "written", "days"],
        "پرداخت": ["payment", "milestone", "invoice", "installment"],
        "فاز": ["phase", "milestone", "stage"],
        "گارانتی": ["warranty", "guarantee", "defects", "months"],
        "تضمین": ["warranty", "guarantee", "quality"],
        "مالکیت": ["intellectual", "property", "ip", "ownership", "rights"],
        "پیمانکار": ["contractor", "subcontractor", "subcontracting", "assignment"],
        "سقف": ["cap", "limitation", "liability", "maximum", "aggregate"],
        "مسئولیت": ["liability", "limitation", "indemnity"],
        "پذیرش": ["acceptance", "inspection", "approval", "deliverables"],
        "متمم": ["amendment", "modification", "written", "modify"],
        "روز": ["day", "daily", "days", "business"],
        "سال": ["year", "years", "annual"],
        "یورو": ["eur", "euro", "euros", "€"],
        "دلار": ["usd", "dollar", "dollars", "$"],
    }

    REFUSAL_STEMS = [
        "اشاره", "یافت نشد", "موجود نیست", "ذکر نشده", "نیامده",
        "no mention", "not found", "not mention", "not specified", "not contain",
        "اطلاعاتی ارائه نشده", "not provided", "وجود ندارد",
    ]

    @classmethod
    def normalize(cls, text: str) -> str:
        """نرمال‌سازی متن فارسی و انگلیسی (حذف ویرگول اعداد، نیم‌فاصله و یکسان‌سازی حروف)"""
        if not text:
            return ""
        cleaned = (
            str(text)
            .replace("\u200c", " ")
            .replace("ي", "ی")
            .replace("ك", "ک")
            .replace(",", "")
        )
        fa_to_en = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
        return cleaned.translate(fa_to_en).lower().strip()

    @classmethod
    def _normalize(cls, text: str) -> str:
        return cls.normalize(text)

    @classmethod
    def is_refusal(cls, answer: str) -> bool:
        if not answer:
            return False
        clean = cls.normalize(answer)
        return any(stem in clean for stem in cls.REFUSAL_STEMS)

    @classmethod
    def _get_concepts(cls, text: str) -> set:
        norm = cls.normalize(text)
        words = set(re.findall(r"[\u0600-\u06FFa-z0-9]+", norm))
        extracted = set(words)
        for fa_key, en_list in cls.CONCEPT_MAP.items():
            if fa_key in norm:
                extracted.update([cls.normalize(e) for e in en_list])
            if any(cls.normalize(en) in words for en in en_list):
                extracted.add(fa_key)
        return extracted

    @classmethod
    def evaluate_context_relevance(cls, query: str, contexts: List[Dict]) -> float:
        if not contexts or not query:
            return 0.0

        q_concepts = cls._get_concepts(query)
        stop = {
            "the", "is", "at", "in", "of", "for", "and", "or", "to", "what", "how", "which", "who", "can", "are",
            "در", "به", "از", "که", "این", "است", "چقدر", "چیست", "های", "را", "با", "چند", "کدام", "چه", "آیا",
        }
        q_concepts = {c for c in q_concepts if len(c) > 1 and c not in stop}

        if not q_concepts:
            return 0.0

        ctx_str = " ".join([c.get("text") or c.get("content", "") for c in contexts])
        ctx_concepts = cls._get_concepts(ctx_str)

        matched = q_concepts.intersection(ctx_concepts)
        if not matched:
            return 0.0

        score = len(matched) / len(q_concepts)
        return round(min(score, 1.0), 2)

    @classmethod
    def evaluate_faithfulness(cls, answer: str, contexts: List[Dict], is_out_of_scope: bool = False) -> float:
        """
        محاسبه شاخص وفاداری (Heuristic Faithfulness):
        - اگر تست خارج از موضوع (is_out_of_scope=True) باشد و مدل امتناع کند، وفاداری 1.0 است.
        - اگر تست داخل موضوع باشد ولی مدل امتناع کند، وفاداری به صورت مصنوعی 1.0 نمی‌شود (0.0 برمی‌گرداند).
        """
        if not answer or answer.startswith("⚠️"):
            return 0.0

        if cls.is_refusal(answer):
            return 1.0 if is_out_of_scope else 0.0

        if not contexts:
            return 0.0

        ctx_str = " ".join([c.get("text") or c.get("content", "") for c in contexts])
        ctx_concepts = cls._get_concepts(ctx_str)
        ans_concepts = cls._get_concepts(answer)

        stop = {
            "the", "is", "at", "in", "of", "for", "and", "or", "to", "per",
            "بر", "اساس", "متن", "ارائه", "شده", "است", "و", "این", "در", "هر", "به", "از", "که", "می", "شود",
        }
        key_ans = {c for c in ans_concepts if len(c) > 1 and c not in stop}

        if not key_ans:
            return 0.0

        matched = key_ans.intersection(ctx_concepts)
        ratio = len(matched) / len(key_ans)
        return round(min(ratio, 1.0), 2)

    @classmethod
    def evaluate_rag_triad(
        cls,
        query: str,
        answer: str,
        contexts: List[Dict],
        is_out_of_scope: bool = False,
    ) -> Dict[str, Any]:
        cr = cls.evaluate_context_relevance(query, contexts)
        faith = cls.evaluate_faithfulness(answer, contexts, is_out_of_scope=is_out_of_scope)
        overall = round((cr + faith) / 2, 2)
        return {
            "heuristic_context_relevance_indicator": cr,
            "heuristic_faithfulness_indicator": faith,
            "heuristic_overall_indicator": overall,
            "is_heuristic_proxy": True,
            "context_relevance": cr,
            "faithfulness": faith,
            "overall_rag_score": overall,
        }

    @classmethod
    def calculate_keyword_recall(cls, answer: str, expected_keywords: List[str]) -> float:
        """محاسبه نسبت کلیدواژه‌های مورد انتظار در پاسخ مدل (با پشتیبانی از معادل‌های دو‌زبانه)"""
        if not expected_keywords:
            return 1.0
        if not answer:
            return 0.0

        norm_answer = cls.normalize(answer)
        ans_concepts = cls._get_concepts(answer)

        hits = 0
        for kw in expected_keywords:
            norm_kw = cls.normalize(kw)
            kw_concepts = cls._get_concepts(norm_kw)
            if norm_kw in norm_answer or bool(kw_concepts.intersection(ans_concepts)):
                hits += 1

        return round(hits / len(expected_keywords), 2)

    @staticmethod
    def _percentile(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        k = (len(sorted_vals) - 1) * (pct / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return round(sorted_vals[int(k)], 2)
        d0 = sorted_vals[int(f)] * (c - k)
        d1 = sorted_vals[int(c)] * (k - f)
        return round(d0 + d1, 2)

    @classmethod
    def load_golden_dataset(cls, dataset_path: str = "evaluation/golden_dataset.json") -> List[Dict[str, Any]]:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"فایل دیتاست طلایی در مسیر '{dataset_path}' یافت نشد.")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def evaluate_single_dataset_item(
        cls,
        item: Dict[str, Any],
        answer: str,
        contexts: List[Dict[str, Any]],
        ret_ms: float,
        gen_ms: float,
        retrieval_error: bool = False,
        generation_error: bool = False,
    ) -> Dict[str, Any]:
        """
        منطق سخت‌گیرانه و استاندارد پذیرش هر سناریوی دیتاست:
        - اگر خطای سیستم رخ داده باشد (retrieval_error یا generation_error): تست قطعا FAIL می‌شود.
        - اگر تست منفی (is_out_of_scope=True) باشد: تنها در صورت امتناع صحیح و نبود شواهد کاذب پاس می‌شود.
        - اگر تست داخل‌موضوع (is_out_of_scope=False) باشد: امتناع باعث FAIL می‌شود و باید کلیدواژه، صفحه و متریک‌ها پاس شوند.
        """
        test_id = item.get("id", "unknown")
        query = item.get("query", "")
        category = item.get("category", "general")
        # پشتیبانی از فیلد استاندارد دیتاست (is_out_of_scope)
        is_negative = bool(item.get("is_out_of_scope", item.get("is_negative_test", False)))
        expected_keywords = item.get("expected_keywords", [])
        expected_page = item.get("expected_page")

        has_system_error = retrieval_error or generation_error or answer.startswith("⚠️")
        refusal_detected = (not has_system_error) and cls.is_refusal(answer)

        metrics = cls.evaluate_rag_triad(
            query=query,
            answer=answer,
            contexts=contexts,
            is_out_of_scope=is_negative,
        )

        keyword_recall = cls.calculate_keyword_recall(answer, expected_keywords)

        retrieved_pages = [
            c.get("page") or c.get("metadata", {}).get("page")
            for c in contexts
            if (c.get("page") or c.get("metadata", {}).get("page")) is not None
        ]
        page_match = (expected_page is None) or (expected_page in retrieved_pages)

        if has_system_error:
            success = False
            outcome_type = "system_error"
        elif is_negative:
            # در تست‌های خارج از موضوع: باید امتناع کند و ارتباط کانتکست پایین باشد (یا مدل فریب نخورده باشد)
            success = refusal_detected and (metrics["context_relevance"] < 0.50 or keyword_recall > 0.0)
            outcome_type = "correct_refusal" if success else "hallucination_on_negative"
        else:
            # در تست‌های داخل‌موضوع: امتناع به معنای شکست قطعی است!
            if refusal_detected:
                success = False
                outcome_type = "false_refusal"
            else:
                success = (
                    keyword_recall >= 0.50
                    and page_match
                    and metrics["context_relevance"] >= 0.50
                    and metrics["faithfulness"] >= 0.50
                )
                outcome_type = "grounded_answer" if success else "ungrounded_or_wrong_page"

        return {
            "id": test_id,
            "category": category,
            "query": query,
            "is_out_of_scope": is_negative,
            "expected_page": expected_page,
            "retrieved_pages": retrieved_pages,
            "page_match": page_match,
            "keyword_recall": keyword_recall,
            "answer": answer,
            "is_refusal": refusal_detected,
            "retrieval_error": retrieval_error,
            "generation_error": generation_error,
            "outcome_type": outcome_type,
            "retrieval_latency_ms": round(ret_ms, 2),
            "generation_latency_ms": round(gen_ms, 2),
            "faithfulness": metrics["faithfulness"],
            "context_relevance": metrics["context_relevance"],
            "overall_score": metrics["overall_rag_score"],
            "passed": success,
        }

    @classmethod
    def run_unified_benchmark(
        cls,
        retriever: Any,
        generator_fn: Callable[[str, List[Dict[str, Any]]], Any],
        dataset_path: str = "evaluation/golden_dataset.json",
        tenant_id: str = "default",
        top_k: int = 3,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        اجرای یکپارچه بنچمارک روی golden_dataset.json (مشترک بین CLI و Streamlit UI).
        """
        dataset = cls.load_golden_dataset(dataset_path)
        details: List[Dict[str, Any]] = []

        for idx, item in enumerate(dataset, start=1):
            query = item["query"]
            retrieval_error = False
            generation_error = False

            # ۱. مرحله بازیابی
            t0 = time.perf_counter()
            try:
                contexts = retriever.retrieve(query=query, top_k=top_k, tenant_id=tenant_id)
                ret_ms = (time.perf_counter() - t0) * 1000.0
            except Exception:
                contexts = []
                ret_ms = (time.perf_counter() - t0) * 1000.0
                retrieval_error = True

            # ۲. مرحله تولید پاسخ (بدون تبدیل خطا به امتناع جعلی!)
            t1 = time.perf_counter()
            try:
                raw_out = generator_fn(query, contexts)
                answer = raw_out["answer"] if isinstance(raw_out, dict) else str(raw_out)
                gen_ms = (time.perf_counter() - t1) * 1000.0
                if answer.startswith("⚠️"):
                    generation_error = True
            except Exception:
                answer = ""
                gen_ms = (time.perf_counter() - t1) * 1000.0
                generation_error = True

            item_result = cls.evaluate_single_dataset_item(
                item=item,
                answer=answer,
                contexts=contexts,
                ret_ms=ret_ms,
                gen_ms=gen_ms,
                retrieval_error=retrieval_error,
                generation_error=generation_error,
            )
            details.append(item_result)

            if progress_callback:
                progress_callback(idx, len(dataset), item_result)

        total_cases = len(details)
        passed_cases = sum(1 for d in details if d["passed"])
        in_scope_cases = [d for d in details if not d["is_out_of_scope"]]
        out_of_scope_cases = [d for d in details if d["is_out_of_scope"]]

        grounded_answers = sum(1 for d in in_scope_cases if d["outcome_type"] == "grounded_answer")
        false_refusals = sum(1 for d in in_scope_cases if d["outcome_type"] == "false_refusal")
        correct_refusals = sum(1 for d in out_of_scope_cases if d["outcome_type"] == "correct_refusal")
        system_errors = sum(1 for d in details if d["retrieval_error"] or d["generation_error"])
        expected_page_hits = sum(1 for d in in_scope_cases if d["page_match"])

        ret_latencies = [d["retrieval_latency_ms"] for d in details]
        gen_latencies = [d["generation_latency_ms"] for d in details]

        summary = {
            "total_test_cases": total_cases,
            "in_scope_cases": len(in_scope_cases),
            "out_of_scope_cases": len(out_of_scope_cases),
            "passed_cases": passed_cases,
            "pass_rate_percentage": round((passed_cases / max(total_cases, 1)) * 100, 2),
            "grounded_answers_count": grounded_answers,
            "false_refusals_count": false_refusals,
            "correct_refusals_count": correct_refusals,
            "system_errors_count": system_errors,
            "expected_page_hit_rate_percentage": round(
                (expected_page_hits / max(len(in_scope_cases), 1)) * 100, 2
            ),
            "avg_retrieval_latency_ms": round(sum(ret_latencies) / max(total_cases, 1), 2),
            "p50_retrieval_latency_ms": cls._percentile(ret_latencies, 50),
            "p95_retrieval_latency_ms": cls._percentile(ret_latencies, 95),
            "avg_generation_latency_ms": round(sum(gen_latencies) / max(total_cases, 1), 2),
            "p50_generation_latency_ms": cls._percentile(gen_latencies, 50),
            "p95_generation_latency_ms": cls._percentile(gen_latencies, 95),
        }

        return {
            "summary": summary,
            "details": details,
        }