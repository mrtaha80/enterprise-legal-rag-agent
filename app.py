import streamlit as st
import tempfile
import os
import time
import html
import json
import hashlib
from pathlib import Path
from src.config import settings
from src.parser import extract_chunks_from_pdf
from src.vector_store import VectorDBService
from src.retriever import HybridRetriever
from src.generator import generate_structured_answer
from src.evaluator import RagEvaluator
from src.security import SecurityGuard

MAX_FILE_SIZE_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024

st.set_page_config(
    page_title="Enterprise Legal RAG System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');

    p, span, div.stMarkdown, input, textarea, label {
        font-family: 'Vazirmatn', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .glass-banner {
        background: radial-gradient(100% 100% at 50% 0%, rgba(30, 58, 138, 0.3) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 18px;
        direction: rtl;
        text-align: right;
    }
    .glass-banner h1 {
        font-size: 24px;
        font-weight: 800;
        color: #f8fafc;
        margin: 0 0 8px 0;
    }
    .glass-banner p {
        font-size: 13.5px;
        color: #94a3b8;
        margin: 0;
    }

    .legal-disclaimer {
        background: rgba(120, 53, 15, 0.25);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-right: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 22px;
        font-size: 13px;
        color: #fde68a;
        direction: rtl;
        text-align: right;
    }

    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 12px;
        direction: rtl;
        text-align: right;
    }

    .glass-answer {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-right: 5px solid #38bdf8;
        border-radius: 12px;
        padding: 22px 24px;
        font-size: 15.5px;
        line-height: 2;
        color: #f1f5f9;
        direction: rtl;
        text-align: right;
    }

    .kpi-card {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 12px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .kpi-title {
        font-size: 12.5px;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 8px;
        line-height: 1.4;
        direction: rtl;
    }
    .kpi-val {
        font-size: 21px;
        font-weight: 800;
        direction: ltr;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_services():
    db = VectorDBService()
    retriever = HybridRetriever(vector_db=db)
    return db, retriever


db_service, retriever = get_services()

st.markdown("""
<div class="glass-banner">
    <h1>⚖️ سامانه سازمانی تحلیل و اعتبارسنجی قراردادها (Persian-First Legal RAG)</h1>
    <p>تفکیک بندمحور (Clause-Aware) • بازیابی ترکیبی (Dense + Normalized Lexical Overlap + RRF) • ایزوله‌سازی داده (Multi-Tenant Data Isolation)</p>
</div>
<div class="legal-disclaimer">
    ⚠️ <strong>سلب مسئولیت حقوقی (Legal Disclaimer):</strong> خروجی‌های این سامانه صرفاً جنبه کمکی و تحلیلی دارد و جایگزین مشاوره حقوقی رسمی یا نظر وکیل دادگستری نمی‌باشد. همچنین متریک‌های لحظه‌ای داشبورد، شاخص‌های اکتشافی (Heuristic Indicators) هستند.
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="section-header">🔐 جداسازی فضای کاری (Demo Shared-PIN Gate)</div>', unsafe_allow_html=True)
    st.caption(
        "توجه معماری: ایزوله‌سازی داده (`tenant_id`) در لایه Qdrant به‌طور کامل پیاده‌سازی شده است، اما این درگاه ورود صرفاً یک مکانیزم نمایشی مبتنی بر PIN مشترک (Demo Shared-PIN) است و جایگزین احراز هویت واقعی کاربر (SSO/OAuth2) نیست."
    )
    username = st.text_input("شناسه فضای کاری (Tenant Identifier):", value="legal_unit_1")

    if not settings.APP_ACCESS_PIN:
        st.error(
            "متغیر محیطی `APP_ACCESS_PIN` تنظیم نشده است! لطفاً برای فعال‌سازی ورود، آن را در فایل `.env` مقداردهی کنید."
        )
        st.stop()

    access_pin = st.text_input("توکن دسترسی محیط دمو (`APP_ACCESS_PIN`):", type="password")
    is_authenticated = (access_pin == settings.APP_ACCESS_PIN) and bool(username.strip())
    tenant_id = hashlib.sha256(username.strip().lower().encode("utf-8")).hexdigest()[:16] if username else "default"

    if not is_authenticated:
        st.warning("برای دسترسی به فضای کاری ایزوله، توکن تعریف‌شده در `APP_ACCESS_PIN` را وارد نمایید.")
        st.stop()
    else:
        st.success(f"فضای کاری فعال: `{username}` (`{tenant_id[:8]}...`)")

    st.divider()
    st.markdown('<div class="section-header">📂 مدیریت مستندات کاربر</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        f"بارگذاری قرارداد (PDF - سقف {settings.MAX_FILE_SIZE_MB}MB):",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("🚀 پردازش و ایندکس در فضای کاری", use_container_width=True):
        if uploaded_files:
            start_ingest = time.perf_counter()
            total_indexed = 0
            progress_bar = st.progress(0, text="شروع پردازش ایمن...")

            for idx, uploaded_file in enumerate(uploaded_files):
                if uploaded_file.size > MAX_FILE_SIZE_BYTES:
                    st.error(f"فایل `{uploaded_file.name}` بیش از سقف مجاز ({settings.MAX_FILE_SIZE_MB}MB) است.")
                    continue

                tmp_path = None
                try:
                    progress_bar.progress(
                        int((idx / len(uploaded_files)) * 100),
                        text=f"تفکیک بندها و اسکن امنیتی `{uploaded_file.name}`...",
                    )
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        while True:
                            buf = uploaded_file.read(8192)
                            if not buf:
                                break
                            tmp.write(buf)
                        tmp_path = tmp.name

                    chunks = extract_chunks_from_pdf(tmp_path, source_name=uploaded_file.name)
                    safe_chunks = SecurityGuard.filter_and_sanitize_chunks(chunks)
                    inserted = db_service.insert_chunks(
                        safe_chunks,
                        tenant_id=tenant_id,
                        clear_existing=True,
                        batch_size=32,
                    )
                    total_indexed += inserted
                except Exception:
                    st.error(f"خطا در پردازش فایل `{uploaded_file.name}`. لطفاً سلامت فایل PDF را بررسی کنید.")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

            progress_bar.progress(100, text="انجام شد!")
            elapsed = round((time.perf_counter() - start_ingest) * 1000, 1)
            st.success(f"{total_indexed} بند حقوقی در فضای کاری `{username}` طی {elapsed} ms ذخیره شد.")
        else:
            st.warning("ابتدا یک فایل PDF انتخاب کنید.")

    tenant_docs = db_service.list_tenant_documents(tenant_id=tenant_id)
    selected_doc_filter = st.selectbox(
        "فیلتر اسناد مجاز شما:",
        options=["همه اسناد مجاز"] + tenant_docs,
    )

tab_qa, tab_bench = st.tabs([
    "🔍 پرسش‌وپاسخ مستند از قراردادها",
    "🧪 اجرای زنده بنچمارک طلایی (11-Case Suite)",
])

with tab_qa:
    query = st.text_input(
        "پرسش حقوقی یا مالی خود را وارد کنید:",
        placeholder="مثلاً: شرایط جریمه تاخیر در تحویل چقدر است؟",
    )

    if st.button("🔍 جستجو و تحلیل قرارداد", use_container_width=True) and query:
        if SecurityGuard.detect_prompt_injection(query):
            st.error("⛔ هشدار امنیتی: الگوی تزریق دستور (Prompt Injection) در پرسش ورودی شناسایی و مسدود شد.")
            st.stop()

        total_start = time.perf_counter()

        with st.spinner("در حال بازیابی مستندات اختصاصی و استنتاج مدل..."):
            t_ret = time.perf_counter()
            contexts = retriever.retrieve(
                query,
                top_k=3,
                tenant_id=tenant_id,
                source_filter=selected_doc_filter,
            )
            ret_ms = round((time.perf_counter() - t_ret) * 1000, 1)

            t_gen = time.perf_counter()
            structured_res = generate_structured_answer(query, contexts)
            answer = structured_res["answer"]
            citations = structured_res["citations"]
            gen_ms = round((time.perf_counter() - t_gen) * 1000, 1)

            evaluation = RagEvaluator.evaluate_rag_triad(query=query, answer=answer, contexts=contexts)
            total_ms = round((time.perf_counter() - total_start) * 1000, 1)

        col_ans, col_chunks = st.columns([6, 4])

        with col_ans:
            st.markdown('<div class="section-header">💡 پاسخ مستند سیستم:</div>', unsafe_allow_html=True)
            safe_escaped_answer = html.escape(answer).replace("\n", "<br>")
            st.markdown(f'<div class="glass-answer">{safe_escaped_answer}</div>', unsafe_allow_html=True)

            if citations:
                st.markdown("#### 📌 استنادهای قطعی مستخرج از متاداده (Deterministic Citations):")
                for cit in citations:
                    st.markdown(
                        f"- 📄 **سند:** `{Path(cit['source']).name}` | **صفحه:** `{cit['page']}` | **شناسه بند:** `{cit['clause_id']}`"
                    )

        with col_chunks:
            st.markdown('<div class="section-header">📑 شواهد و بندهای مرتبط:</div>', unsafe_allow_html=True)
            if contexts:
                for idx, c in enumerate(contexts):
                    score = c.get("rerank_score", 0.0)
                    meta = c.get("metadata", {})
                    clean_name = Path(meta.get("source", "unknown")).name
                    page_num = meta.get("page", 1)
                    clause_id = meta.get("clause_id", "ثبت‌نشده")
                    text_content = c.get("content", "")

                    with st.expander(f"مستند {idx + 1}: {clean_name} — صفحه {page_num}"):
                        st.write(text_content)
                        st.caption(f"شناسه بند: `{clause_id}` | امتیاز بازچینی: `{score:.4f}`")
            else:
                st.warning("هیچ قطعه مرتبطی در فضای کاری شما یافت نشد.")

        st.markdown("---")
        st.markdown(
            '<div class="section-header">📊 داشبورد شاخص‌های اکتشافی (Heuristic Proxy Indicators)</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">زمان کل پاسخ‌دهی</div>
                <div class="kpi-val" style="color: #60a5fa;">{total_ms} ms</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">جستجو و بازچینی</div>
                <div class="kpi-val" style="color: #fbbf24;">{ret_ms} ms</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">استنتاج مدل (LLM)</div>
                <div class="kpi-val" style="color: #c084fc;">{gen_ms} ms</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            rel_percent = int(evaluation["heuristic_context_relevance_indicator"] * 100)
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Heuristic Relevance Indicator</div>
                <div class="kpi-val" style="color: #34d399;">{rel_percent}%</div>
            </div>
            """, unsafe_allow_html=True)

        with c5:
            faith_percent = int(evaluation["heuristic_faithfulness_indicator"] * 100)
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Heuristic Faithfulness Indicator</div>
                <div class="kpi-val" style="color: #4ade80;">{faith_percent}%</div>
            </div>
            """, unsafe_allow_html=True)

with tab_bench:
    st.markdown("### 🧪 اجرای عملیاتی مجموعه بنچمارک طلایی (11-Case Multi-Scenario Suite)")
    st.caption(
        "این مجموعه شامل سناریوهای فارسی، انگلیسی، استخراج عددی، چندبندی، چندصفحه‌ای، تضاد الحاقیه، خارج از موضوع و تست دفاع در برابر Prompt Injection است."
    )

    if st.button("▶️ اجرای زنده بنچمارک End-to-End و تولید گزارش", use_container_width=True):
        with st.spinner("در حال اجرای ۱۱ سناریوی مرجع روی Qdrant و LLM..."):
            report = RagEvaluator.run_golden_benchmark_suite(
                db_service=db_service,
                retriever=retriever,
                generator_fn=generate_structured_answer,
                use_live_retrieval=True,
            )

        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("نرخ قبولی (Pass Rate)", f"{int(report['pass_rate'] * 100)}%", f"{report['passed_cases']}/{report['total_cases']}")
        b2.metric("میانگین Keyword Recall", f"{int(report['avg_keyword_recall'] * 100)}%")
        b3.metric("دقت استناد (Citation)", f"{int(report['avg_citation_accuracy'] * 100)}%")
        b4.metric("میانگین Retrieval", f"{report['avg_retrieval_ms']} ms")
        b5.metric("میانگین Generation", f"{report['avg_generation_ms']} ms")

        st.download_button(
            label="📥 دانلود گزارش کامل بنچمارک (JSON)",
            data=json.dumps(report, ensure_ascii=False, indent=2),
            file_name="golden_benchmark_report.json",
            mime="application/json",
        )

        st.markdown("#### 📋 جزئیات نتایج هر سناریو:")
        for res in report["results"]:
            status_icon = "✅ PASSED" if res["passed"] else "❌ FAILED"
            with st.expander(f"{status_icon} | [{res['category']}] {res['case_id']} — {res['query']}"):
                st.markdown(f"**پاسخ تولیدشده:**\n> {res['generated_answer']}")
                st.write({
                    "keyword_recall": res["keyword_recall"],
                    "citation_accuracy": res["citation_accuracy"],
                    "retrieval_ms": res["retrieval_ms"],
                    "generation_ms": res["generation_ms"],
                    "passed": res["passed"],
                })