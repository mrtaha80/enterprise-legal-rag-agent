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

# استایل ایزوله‌شده بدون تداخل با فونت آیکون‌ها و المان‌های داخلی استریم‌لیت
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');

    /* اعمال فونت منحصراً بر روی عناصر متنی استاندارد */
    p, span, div.stMarkdown, input, textarea, label {
        font-family: 'Vazirmatn', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* بنر اصلی Glassmorphism */
    .glass-banner {
        background: radial-gradient(100% 100% at 50% 0%, rgba(30, 58, 138, 0.25) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
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

    /* جعبه پاسخ سیستم */
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
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }

    /* کارت‌های داشبورد متریک */
    .kpi-card {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 12px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.3);
    }
    .kpi-title {
        font-size: 12px;
        font-weight: 500;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    .kpi-val {
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.5px;
        direction: ltr;
        display: inline-block;
    }

    /* کارت شواهد و بندها */
    .clause-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
        direction: rtl;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_services():
    db = VectorDBService()
    retriever = HybridRetriever()
    return db, retriever

db_service, retriever = get_services()

# بنر معرفی
st.markdown("""
<div class="glass-banner">
    <h1>⚖️ سامانه سازمانی تحلیل و اعتبارسنجی قراردادها (Legal RAG)</h1>
    <p>موتور استنتاج مبتنی بر تفکیک بندمحور (Clause-Aware) • رتبه‌بندی مجدد دو‌سطحی • پایش تاخیر و ارزیابی کمی RAG Triad</p>
</div>
""", unsafe_allow_html=True)

# نوار سایدبار
with st.sidebar:
    st.subheader("📂 مدیریت مستندات")
    uploaded_files = st.file_uploader(
        "بارگذاری فایل‌های قرارداد (PDF):",
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
            st.success(f"{total_chunks} بند حقوقی در {elapsed} ms ذخیره شد.")
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

# ورودی پرسش
query = st.text_input(
    "پرسش حقوقی یا مالی خود را وارد کنید:",
    placeholder="مثلاً: شرایط جریمه تاخیر در تحویل چقدر است؟"
)

if st.button("🔍 جستجو و تحلیل قرارداد", use_container_width=True) and query:
    total_start = time.perf_counter()
    
    with st.spinner("در حال بازیابی مستندات و استنتاج مدل..."):
        # ۱. جستجو و بازچینی
        t_ret = time.perf_counter()
        contexts = retriever.retrieve(query, top_k=3)
        ret_ms = round((time.perf_counter() - t_ret) * 1000, 1)

        # ۲. پاسخ‌دهی LLM
        t_gen = time.perf_counter()
        answer = generate_answer(query, contexts)
        gen_ms = round((time.perf_counter() - t_gen) * 1000, 1)
        
        # ۳. ارزیابی سلامت و وفاداری پاسخ
        evaluation = RagEvaluator.evaluate_rag_triad(query=query, answer=answer, contexts=contexts)
        total_ms = round((time.perf_counter() - total_start) * 1000, 1)

    col_ans, col_chunks = st.columns([6, 4])
    
    with col_ans:
        st.subheader("💡 پاسخ مستند سیستم:")
        formatted_answer = answer.replace("\n", "<br>")
        st.markdown(f'<div class="glass-answer">{formatted_answer}</div>', unsafe_allow_html=True)

    with col_chunks:
        st.subheader("📑 شواهد و بندهای مرتبط:")
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

    # داشبورد متریک‌های سازمانی
    st.subheader("📊 داشبورد رصدپذیری و سنجش کیفیت (Observability & Evaluation)")
    
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
            <div class="kpi-title">بازیابی و ریرنک</div>
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
            <div class="kpi-title">ارتباط شواهد (Relevance)</div>
            <div class="kpi-val" style="color: #34d399;">{rel_percent}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c5:
        faith_percent = int(evaluation["faithfulness"] * 100)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">وفاداری به متن (Faithfulness)</div>
            <div class="kpi-val" style="color: #4ade80;">{faith_percent}%</div>
        </div>
        """, unsafe_allow_html=True)