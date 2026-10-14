import json
import time
from pathlib import Path
from typing import Dict, List
from src.retriever import HybridRetriever
from src.generator import generate_answer
from src.evaluator import RagEvaluator


def load_golden_dataset(dataset_path: str = "evaluation/golden_dataset.json") -> List[Dict]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"دیتاست در مسیر {dataset_path} یافت نشد.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_benchmark(dataset_path: str = "evaluation/golden_dataset.json"):
    print("=" * 60)
    print("⚖️  آغاز اجرای بنچمارک جامع RAG Triad روی Golden Dataset")
    print("=" * 60)

    dataset = load_golden_dataset(dataset_path)
    retriever = HybridRetriever()

    results = []
    total_retrieval_ms = 0.0
    total_generation_ms = 0.0
    passed_cases = 0

    for idx, item in enumerate(dataset, start=1):
        test_id = item.get("id", f"case_{idx}")
        query = item["query"]
        category = item.get("category", "General")
        is_negative = item.get("is_negative_test", False)

        print(f"\n[{idx}/{len(dataset)}] بررسی {test_id} ({category}): {query}")

        # ۱. بازیابی و ریرنکینگ
        t0 = time.perf_counter()
        contexts = retriever.retrieve(query, top_k=3)
        ret_ms = round((time.perf_counter() - t0) * 1000, 2)
        total_retrieval_ms += ret_ms

        # ۲. تولید پاسخ
        t1 = time.perf_counter()
        answer = generate_answer(query, contexts)
        gen_ms = round((time.perf_counter() - t1) * 1000, 2)
        total_generation_ms += gen_ms

        # ۳. ارزیابی کیفی
        metrics = RagEvaluator.evaluate_rag_triad(query=query, answer=answer, contexts=contexts)

        # ارزیابی تست منفی (ضد-توهم)
        # ارزیابی تست منفی (ضد-توهم)
        if is_negative:
            refusal_markers = [
                "یافت نشد", "اشاره‌ای نشده", "no mention", "not found", 
                "در متن نیامده", "ذکر نشده", "موجود نیست", "does not contain", 
                "not provide", "not specified"
            ]
            success = any(m in answer.lower() for m in refusal_markers) or metrics["overall_rag_score"] >= 0.70
        else:
            success = metrics["faithfulness"] >= 0.70 and metrics["context_relevance"] >= 0.60

        if success:
            passed_cases += 1
            status_symbol = "✅ PASSED"
        else:
            status_symbol = "❌ FAILED"

        print(f"   وضعیت: {status_symbol} | تاخیر: بازیابی {ret_ms}ms, مدل {gen_ms}ms")
        print(f"   امتیازات: وفاداری {int(metrics['faithfulness']*100)}% | ارتباط {int(metrics['context_relevance']*100)}%")

        results.append({
            "id": test_id,
            "category": category,
            "query": query,
            "answer": answer,
            "retrieval_latency_ms": ret_ms,
            "generation_latency_ms": gen_ms,
            "faithfulness": metrics["faithfulness"],
            "context_relevance": metrics["context_relevance"],
            "overall_score": metrics["overall_rag_score"],
            "passed": success,
        })

    # خلاصه گزارش
    total_tests = len(dataset)
    pass_rate = round((passed_cases / total_tests) * 100, 2)
    avg_ret_ms = round(total_retrieval_ms / total_tests, 2)
    avg_gen_ms = round(total_generation_ms / total_tests, 2)

    report = {
        "summary": {
            "total_test_cases": total_tests,
            "passed_cases": passed_cases,
            "pass_rate_percentage": pass_rate,
            "avg_retrieval_latency_ms": avg_ret_ms,
            "avg_generation_latency_ms": avg_gen_ms,
        },
        "details": results,
    }

    report_path = Path("evaluation/benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("📊 گزارش پایانی بنچمارک سازمانی")
    print(f"تعداد کل تست‌ها: {total_tests}")
    print(f"تست‌های موفق: {passed_cases} ({pass_rate}%)")
    print(f"میانگین تاخیر بازیابی: {avg_ret_ms} ms")
    print(f"میانگین تاخیر مدل زبانی: {avg_gen_ms} ms")
    print(f"گزارش تفصیلی در مسیر ذخیره شد: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()