import json
import os
from pathlib import Path

print("⏳ در حال بارگذاری کتابخانه‌های هوش مصنوعی و موتور ارزیابی واحد...", flush=True)

from src.vector_store import VectorDBService
from src.retriever import HybridRetriever
from src.generator import generate_answer
from src.evaluator import RagEvaluator


def print_step_progress(idx: int, total: int, res: dict):
    status_symbol = "✅ PASSED" if res["passed"] else "❌ FAILED"
    scope_tag = "OUT-OF-SCOPE" if res["is_out_of_scope"] else "IN-SCOPE"
    print(f"[{idx}/{total}] {res['id']} ({res['category']} | {scope_tag}): {res['query']}")
    print(
        f"   وضعیت: {status_symbol} ({res['outcome_type']}) | "
        f"بازیابی: {res['retrieval_latency_ms']}ms | مدل: {res['generation_latency_ms']}ms"
    )
    print(
        f"   تطابق صفحه: {res['page_match']} (انتظار: {res['expected_page']}، بازیابی: {res['retrieved_pages']}) | "
        f"Keyword Recall: {int(res['keyword_recall'] * 100)}% | "
        f"ارتباط: {int(res['context_relevance'] * 100)}% | وفاداری: {int(res['faithfulness'] * 100)}%\n",
        flush=True,
    )


def run_benchmark(
    dataset_path: str = "evaluation/golden_dataset.json",
    report_output_path: str = "evaluation/benchmark_report.json",
    tenant_id: str = None,
):
    print("\n" + "=" * 72)
    print("⚖️  آغاز اجرای بنچمارک سخت‌گیرانه (Strict Unified Benchmark) روی Golden Dataset")
    print("=" * 72, flush=True)

    active_tenant = tenant_id or os.getenv("BENCHMARK_TENANT_ID", "default")
    db = VectorDBService()
    retriever = HybridRetriever(vector_db=db)
    print(f"✅ سرویس‌ها آماده شدند (Tenant: '{active_tenant}'). شروع ارزیابی...\n", flush=True)

    report = RagEvaluator.run_unified_benchmark(
        retriever=retriever,
        generator_fn=generate_answer,
        dataset_path=dataset_path,
        tenant_id=active_tenant,
        top_k=3,
        progress_callback=print_step_progress,
    )

    report_path = Path(report_output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    s = report["summary"]
    print("=" * 72)
    print("📊 خلاصه مهندسی بنچمارک (Strict Evaluation Summary):")
    print(f"• کل سناریوها: {s['total_test_cases']} (داخل‌موضوع: {s['in_scope_cases']} | خارج‌ازموضوع: {s['out_of_scope_cases']})")
    print(f"• تست‌های قبول‌شده واقعی: {s['passed_cases']} از {s['total_test_cases']} ({s['pass_rate_percentage']}%)")
    print(f"• پاسخ‌های مستند موفق (Grounded Answers): {s['grounded_answers_count']} از {s['in_scope_cases']}")
    print(f"• امتناع‌های غلط روی سوالات معتبر (False Refusals): {s['false_refusals_count']}")
    print(f"• امتناع‌های درست روی سوالات نامرتبط (Correct Refusals): {s['correct_refusals_count']} از {s['out_of_scope_cases']}")
    print(f"• خطاهای سیستمی (System Errors): {s['system_errors_count']}")
    print(f"• نرخ بازیابی صفحه مورد انتظار (Expected Page Hit Rate): {s['expected_page_hit_rate_percentage']}%")
    print(
        f"• تاخیر بازیابی (Retrieval): Avg={s['avg_retrieval_latency_ms']}ms | "
        f"P50={s['p50_retrieval_latency_ms']}ms | P95={s['p95_retrieval_latency_ms']}ms"
    )
    print(
        f"• تاخیر تولید پاسخ (Generation): Avg={s['avg_generation_latency_ms']}ms | "
        f"P50={s['p50_generation_latency_ms']}ms | P95={s['p95_generation_latency_ms']}ms"
    )
    print(f"📁 گزارش تفصیلی ذخیره شد در: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    run_benchmark()