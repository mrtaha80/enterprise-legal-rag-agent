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
    layout="wide"
)

# استایل‌بندی سفارشی با فونت استاندارد Vazirmatn و تم سازمانی تیره
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');
    
    * {
        font-family: 'Vazirmatn', sans-serif !important;
    }
    
    .stApp {
        direction: rtl;
        text-align: right;
        background-color: #0d1117;
    }
    
    /* سربرگ مدرن */
    .hero-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 24px;
        font-weight: 800;
        color: #f1f5f9;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 14px;
        color: #94a3b8;
    }

    /* کادر پاسخ سیستم */
    .answer-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-right: 4px solid #38bdf8;
        border-radius: 10px;
        padding: 20px;
        font-size: 15.5px;
        line-height: 1.9;
        color: #e6edf3;
        margin-bottom: 15px;
    }

    /* کارت‌های متریک داشبورد */
    .metric-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-label {
        font-size: 12.5px;
        color: #8b949e;
        margin-bottom: 6px;
    }
    .metric-val {
        font-size: 20px;
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
<div class="hero-banner">
    <div class="hero-title">⚖️ سامانه سازمانی تحلیل و اعتبارسنجی قراردادها (Legal RAG)</div>
    <div class="hero-subtitle">بازیابی دوسطحی برداری و متقاطع • خردایش بندمحور • پایش تاخیر و ارزیابی کیفیت RAG Triad</div>
</div>
""", unsafe_allow_html=True)

# نوار کناری
with st.sidebar:
    st.header("📂 مدیریت مستندات")
    uploaded_files = st.file_uploader(
        "بارگذاری اسناد قرارداد یا مالی (PDF):",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if st.button("🚀 پردازش و ایندکس برداری"):
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
    - **Vector DB:** Qdrant (HNSW / Cosine)
    - **Reranker:** Cross-Encoder (MS-Marco)
    - **LLM Engine:** Llama 3.2
    """)

# ورودی پرسش
query = st.text_input(
    "پرسش حقوقی یا مالی خود را مطرح کنید:",
    placeholder="مثلاً: شرایط جریمه دیرکرد تحویل فازها چیست؟"
)

if st.button("🔍 جستجو و تحلیل قرارداد") and query:
    total_start = time.perf_counter()
    
    with st.spinner("در حال بازیابی، بازچینی و استنتاج مدل..."):
        # بازیابی
        t_ret = time.perf_counter()
        contexts = retriever.retrieve(query, top_k=3)
        ret_ms = round((time.perf_counter() - t_ret) * 1000, 1)

        # استنتاج
        t_gen = time.perf_counter()
        answer = generate_answer(query, contexts)
        gen_ms = round((time.perf_counter() - t_gen) * 1000, 1)
        
        # ارزیابی خودکار
        evaluation = RagEvaluator.evaluate_rag_triad(query=query, answer=answer, contexts=contexts)
        total_ms = round((time.perf_counter() - total_start) * 1000, 1)

    # دو ستون اصلی نتایج
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
                # نمایش صرفاً نام تمیز فایل بدون مسیر ویندوز
                clean_name = Path(c["source"]).name
                clause_id = c.get("metadata", {}).get("clause_id", "ثبت‌نشده")
                
                with st.expander(f"مستند {idx+1}: {clean_name} (صفحه {c['page']})"):
                    st.write(c["text"])
                    st.caption(f"شناسه بند: `{clause_id}` | امتیاز بازچینی: `{score:.4f}`")
        else:
            st.warning("هیچ قطعه مرتبطی یافت نشد.")

    st.markdown("---")

    # داشبورد متریک‌ها و مانیتورینگ
    st.subheader("📊 داشبورد رصدپذیری و سنجش کیفیت (Observability & Evaluation)")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">زمان کل پاسخ‌دهی</div>
            <div class="metric-val" style="color: #60a5fa;">{total_ms} ms</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">جستجو و بازچینی</div>
            <div class="metric-val" style="color: #fbbf24;">{ret_ms} ms</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">استنتاج مدل (LLM)</div>
            <div class="metric-val" style="color: #c084fc;">{gen_ms} ms</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c4:
        rel_percent = int(evaluation["context_relevance"] * 100)
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">ارتباط شواهد</div>
            <div class="metric-val" style="color: #34d399;">{rel_percent}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c5:
        faith_percent = int(evaluation["faithfulness"] * 100)
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">وفاداری به متن</div>
            <div class="metric-val" style="color: #4ade80;">{faith_percent}%</div>
        </div>
        """, unsafe_allow_html=True)