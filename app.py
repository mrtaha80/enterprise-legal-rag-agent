import streamlit as st
import tempfile
import os
import time
from pathlib import Path
from src.parser import extract_chunks_from_pdf
from src.vector_store import VectorDBService
from src.retriever import HybridRetriever
from src.generator import generate_answer
from src.evaluator import RagEvaluator

st.set_page_config(
    page_title="Enterprise Legal RAG System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# استایل ایزوله با هدایت کاراکتری دقیق
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');

    p, span, div.stMarkdown, input, textarea, label {
        font-family: 'Vazirmatn', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* بنر اصلی با گرادیان سرمه‌ای تیره */
    .glass-banner {
        background: radial-gradient(100% 100% at 50% 0%, rgba(30, 58, 138, 0.3) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 25px;
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

    /* تیترهای بخش‌ها با تراز کاملاً راست */
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 12px;
        direction: rtl;
        text-align: right;
    }

    /* کادر پاسخ سیستم */
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

    /* کارت‌های متریک داشبورد پایین صفحه */
    .kpi-card {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 12px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .kpi-title {
        font-size: 13px;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 8px;
        line-height: 1.4;
        direction: rtl;
    }
    .kpi-val {
        font-size: 22px;
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
# بنر اصلی
st.markdown("""
<div class="glass-banner">
    <h1>⚖️ سامانه سازمانی تحلیل و اعتبارسنجی قراردادها (Legal RAG)</h1>
    <p>موتور استنتاج مبتنی بر تفکیک بندمحور (Clause-Aware) • رتبه‌بندی مجدد دو‌سطحی • پایش تاخیر و ارزیابی کمی RAG Triad</p>
</div>
""", unsafe_allow_html=True)

# نوار سایدبار
with st.sidebar:
    st.markdown('<div class="section-header">📂 مدیریت مستندات</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "بارگذاری فایل‌های قرارداد (PDF):",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if st.button("🚀 پردازش و ایندکس برداری", use_container_width=True):
        if uploaded_files:
            all_new_chunks = []
            start_ingest = time.perf_counter()
            with st.spinner("در حال تفکیک مواد قانونی و بردارسازی..."):
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    
                    chunks = extract_chunks_from_pdf(tmp_path, source_name=uploaded_file.name)
                    all_new_chunks.extend(chunks)
                    os.remove(tmp_path)
                
                # پاکسازی نسخه‌های تکراری قبلی و درج یکپارچه بندهای جدید
                db_service.insert_chunks(all_new_chunks, clear_existing=True)
            
            elapsed = round((time.perf_counter() - start_ingest) * 1000, 1)
            st.success(f"{len(all_new_chunks)} بند حقوقی یکتا در {elapsed} ms ذخیره شد.")
        else:
            st.warning("ابتدا یک فایل PDF انتخاب کنید.")
            
    st.divider()
    st.markdown("#### ⚙️ معماری خط لوله (Pipeline)")
    st.markdown("""
    - **Chunking:** Clause-Aware Regex
    - **Vector Store:** Qdrant (Cosine)
    - **Reranker:** Cross-Encoder
    - **LLM Engine:** Llama 3.2
    """)

# فرم ورودی
query = st.text_input(
    "پرسش حقوقی یا مالی خود را وارد کنید:",
    placeholder="مثلاً: شرایط جریمه تاخیر در تحویل چقدر است؟"
)

if st.button("🔍 جستجو و تحلیل قرارداد", use_container_width=True) and query:
    total_start = time.perf_counter()
    
    with st.spinner("در حال بازیابی مستندات و استنتاج مدل..."):
        # ۱. بازیابی و ریرنکینگ
        t_ret = time.perf_counter()
        contexts = retriever.retrieve(query, top_k=3)
        ret_ms = round((time.perf_counter() - t_ret) * 1000, 1)

        # ۲. پاسخ‌دهی LLM
        t_gen = time.perf_counter()
        answer = generate_answer(query, contexts)
        gen_ms = round((time.perf_counter() - t_gen) * 1000, 1)
        
        # ۳. ارزیابی کیفی RAG Triad
        evaluation = RagEvaluator.evaluate_rag_triad(query=query, answer=answer, contexts=contexts)
        total_ms = round((time.perf_counter() - total_start) * 1000, 1)

    col_ans, col_chunks = st.columns([6, 4])
    
    with col_ans:
        st.markdown('<div class="section-header">💡 پاسخ مستند سیستم:</div>', unsafe_allow_html=True)
        formatted_answer = answer.replace("\n", "<br>")
        st.markdown(f'<div class="glass-answer">{formatted_answer}</div>', unsafe_allow_html=True)

    with col_chunks:
        st.markdown('<div class="section-header">📑 شواهد و بندهای مرتبط:</div>', unsafe_allow_html=True)
        if contexts:
            for idx, c in enumerate(contexts):
                score = c.get("rerank_score", 0.0)
                clean_name = Path(c["source"]).name
                clause_id = c.get("metadata", {}).get("clause_id", "ثبت‌نشده")
                text_content = c.get("text") or c.get("content") or ""
                
                with st.expander(f"مستند {idx+1}: {clean_name} — صفحه {c['page']}"):
                    st.write(text_content)
                    st.caption(f"شناسه بند: `{clause_id}` | امتیاز بازچینی: `{score:.4f}`")
        else:
            st.warning("هیچ قطعه مرتبطی یافت نشد.")

    st.markdown("---")

    # داشبورد متریک‌ها
    st.markdown('<div class="section-header">📊 داشبورد رصدپذیری و سنجش کیفیت (Observability & Evaluation)</div>', unsafe_allow_html=True)
    
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
        rel_percent = int(evaluation["context_relevance"] * 100)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">ارتباط شواهد</div>
            <div class="kpi-val" style="color: #34d399;">{rel_percent}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c5:
        faith_percent = int(evaluation["faithfulness"] * 100)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">وفاداری به متن</div>
            <div class="kpi-val" style="color: #4ade80;">{faith_percent}%</div>
        </div>
        """, unsafe_allow_html=True)