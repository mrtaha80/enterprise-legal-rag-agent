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
    page_title="سامانه سازمانی تحلیل و اعتبارسنجی قراردادها",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# حل کامل باگ سایدبار و اصلاح جهت‌دهی و آیکون‌ها
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');

    /* اعمال فونت روی متون بدون ایجاد اختلال در آیکون‌ها */
    html, body, p, span, h1, h2, h3, h4, h5, h6, input, textarea, button {
        font-family: 'Vazirmatn', -apple-system, sans-serif !important;
    }

    /* جلوگیری کامل از خراب شدن آیکون‌های پیش‌فرض Streamlit */
    [data-testid="stIcon"], i, svg, [class*="material-icons"] {
        font-family: inherit !important;
    }

    /* رفع باگ عمودی شدن محتوای سایدبار هنگام جمع شدن */
    [data-testid="stSidebar"] {
        overflow-x: hidden !important;
    }
    [data-testid="stSidebar"][aria-expanded="false"] * {
        display: none !important;
    }

    /* جهت‌بندی راست‌به‌چپ برای بدنه */
    .stApp {
        direction: rtl;
        text-align: right;
        background-color: #0b0f17;
    }

    /* کادر هدر صفحه */
    .custom-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 24px;
    }
    .custom-header h2 {
        color: #f8fafc;
        font-size: 22px;
        margin-bottom: 6px;
    }
    .custom-header p {
        color: #94a3b8;
        font-size: 13.5px;
        margin: 0;
    }

    /* کادر پاسخ مدل */
    .answer-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-right: 4px solid #38bdf8;
        border-radius: 10px;
        padding: 20px;
        color: #f3f4f6;
        line-height: 1.85;
        font-size: 15px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }

    /* باکس‌های داشبورد متریک */
    .stat-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 16px 10px;
        text-align: center;
    }
    .stat-label {
        font-size: 12px;
        color: #9ca3af;
        margin-bottom: 6px;
    }
    .stat-number {
        font-size: 22px;
        font-weight: 700;
        direction: ltr;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_services():
    db = VectorDBService()
    retriever = HybridRetriever()
    return db, retriever

db_service, retriever = get_services()

# سربرگ صفحه
st.markdown("""
<div class="custom-header">
    <h2>⚖️ سامانه سازمانی تحلیل و اعتبارسنجی قراردادها (Legal RAG)</h2>
    <p>بازیابی دوسطحی برداری و متقاطع • خردایش بندمحور • پایش تاخیر و ارزیابی کیفیت RAG Triad</p>
</div>
""", unsafe_allow_html=True)

# نوار سایدبار
with st.sidebar:
    st.header("📂 مدیریت مستندات")
    uploaded_files = st.file_uploader(
        "بارگذاری اسناد حقوقی و مالی (PDF):",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if st.button("🚀 پردازش و ایندکس برداری", use_container_width=True):
        if uploaded_files:
            total_chunks = 0
            start_ingest = time.perf_counter()
            with st.spinner("در حال تفکیک مواد قانونی و بردارسازی..."):
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    
                    chunks = extract_chunks_from_pdf(tmp_path, source_name=uploaded_file.name)
                    db_service.insert_chunks(chunks)
                    total_chunks += len(chunks)
                    os.remove(tmp_path)
            
            elapsed = round((time.perf_counter() - start_ingest) * 1000, 1)
            st.success(f"{total_chunks} بند حقوقی در {elapsed} ms ایندکس شد.")
        else:
            st.warning("ابتدا یک فایل PDF انتخاب کنید.")
            
    st.divider()
    st.markdown("#### ⚙️ معماری سامانه")
    st.markdown("""
    - **Chunking:** Clause-Aware Regex
    - **Vector DB:** Qdrant (Cosine)
    - **Reranker:** Cross-Encoder
    - **LLM Engine:** Llama 3.2
    """)

# ورودی پرسش
query = st.text_input(
    "پرسش حقوقی یا مالی خود را وارد کنید:",
    placeholder="مثلاً: شرایط جریمه تاخیر در تحویل چقدر است؟"
)

if st.button("🔍 جستجو و تحلیل قرارداد", use_container_width=True) and query:
    total_start = time.perf_counter()
    
    with st.spinner("در حال بازیابی مستندات و استنتاج مدل..."):
        t_ret = time.perf_counter()
        contexts = retriever.retrieve(query, top_k=3)
        ret_ms = round((time.perf_counter() - t_ret) * 1000, 1)

        t_gen = time.perf_counter()
        answer = generate_answer(query, contexts)
        gen_ms = round((time.perf_counter() - t_gen) * 1000, 1)
        
        evaluation = RagEvaluator.evaluate_rag_triad(query=query, answer=answer, contexts=contexts)
        total_ms = round((time.perf_counter() - total_start) * 1000, 1)

    col_ans, col_chunks = st.columns([6, 4])
    
    with col_ans:
        st.subheader("💡 پاسخ مستند سیستم:")
        formatted_answer = answer.replace("\n", "<br>")
        st.markdown(f'<div class="answer-card">{formatted_answer}</div>', unsafe_allow_html=True)

    with col_chunks:
        st.subheader("📑 شواهد و بندهای مرتبط:")
        if contexts:
            for idx, c in enumerate(contexts):
                score = c.get("rerank_score", 0.0)
                clean_name = Path(c["source"]).name
                clause_id = c.get("metadata", {}).get("clause_id", "ثبت‌نشده")
                text_content = c.get("text") or c.get("content") or ""
                
                with st.expander(f"مستند {idx+1}: {clean_name} (صفحه {c['page']})"):
                    st.write(text_content)
                    st.caption(f"شناسه بند: `{clause_id}` | امتیاز بازچینی: `{score:.4f}`")
        else:
            st.warning("هیچ قطعه مرتبطی یافت نشد.")

    st.markdown("---")

    # داشبورد متریک‌های کیفیت و تاخیر
    st.subheader("📊 داشبورد رصدپذیری و سنجش کیفیت (Observability & Evaluation)")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">زمان کل پاسخ‌دهی</div>
            <div class="stat-number" style="color: #60a5fa;">{total_ms} ms</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">جستجو و بازچینی</div>
            <div class="stat-number" style="color: #fbbf24;">{ret_ms} ms</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">استنتاج مدل (LLM)</div>
            <div class="stat-number" style="color: #c084fc;">{gen_ms} ms</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c4:
        rel_percent = int(evaluation["context_relevance"] * 100)
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">ارتباط شواهد</div>
            <div class="stat-number" style="color: #34d399;">{rel_percent}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c5:
        faith_percent = int(evaluation["faithfulness"] * 100)
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">وفاداری به متن</div>
            <div class="stat-number" style="color: #4ade80;">{faith_percent}%</div>
        </div>
        """, unsafe_allow_html=True)