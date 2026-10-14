import streamlit as st
import tempfile
import os
import time
from src.parser import extract_chunks_from_pdf
from src.vector_store import VectorDBService
from src.retriever import HybridRetriever
from src.generator import generate_answer
from src.evaluator import RagEvaluator

st.set_page_config(
    page_title="Enterprise Legal & Financial RAG Agent",
    page_icon="⚖️",
    layout="wide"
)

# بارگذاری سرویس‌ها به صورت کش شده برای افزایش بازدهی
@st.cache_resource
def get_services():
    db = VectorDBService()
    retriever = HybridRetriever()
    return db, retriever

db_service, retriever = get_services()

st.title("⚖️ سامانه سازمانی تحلیل و اعتبارسنجی قراردادها (Legal RAG)")
st.caption("سیستم هوشمند بازیابی مبتنی بر Clause-Aware Chunking، رتبه‌بندی مجدد و ارزیابی کمی RAG Triad")

# سایدبار: مدیریت اسناد
with st.sidebar:
    st.header("📂 مدیریت و بارگذاری اسناد")
    uploaded_files = st.file_uploader(
        "فایل‌های PDF قرارداد یا اسناد مالی را انتخاب کنید:",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if st.button("🚀 پردازش، تکه‌بندی و ایندکس برداری"):
        if uploaded_files:
            total_chunks = 0
            start_ingest = time.perf_counter()
            with st.spinner("در حال پردازش هوشمند بندها و استخراج وکتورها..."):
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    
                    chunks = extract_chunks_from_pdf(tmp_path, source_name=uploaded_file.name)
                    db_service.insert_chunks(chunks)
                    total_chunks += len(chunks)
                    os.remove(tmp_path)
            
            ingest_latency = round((time.perf_counter() - start_ingest) * 1000, 2)
            st.success(f"تعداد {total_chunks} چانک بندمحور در {ingest_latency} میلی‌ثانیه با موفقیت ایندکس شد.")
        else:
            st.warning("ابتدا حداقل یک فایل PDF انتخاب کنید.")
            
    st.divider()
    st.markdown("### ⚙️ معماری سامانه")
    st.markdown("""
    - **Chunking:** Clause-Aware Regex
    - **Vector DB:** Qdrant (HNSW / Cosine)
    - **Reranker:** Cross-Encoder
    - **Evaluation:** Faithfulness & Relevance
    """)

# بخش اصلی: پرسش و تحلیل
query = st.text_input(
    "پرسش حقوقی یا مالی خود را مطرح کنید:",
    placeholder="مثلاً: شرایط جریمه دیرکرد تحویل فازها چیست؟"
)

if st.button("🔍 جستجو و استنتاج مستند") and query:
    total_start = time.perf_counter()
    
    with st.spinner("در حال بازیابی، بازچینی و تولید پاسخ مستند..."):
        # ۱. بازیابی و ریرنکینگ چانک‌ها
        t_ret_start = time.perf_counter()
        contexts = retriever.retrieve(query, top_k=3)
        retrieval_ms = round((time.perf_counter() - t_ret_start) * 1000, 2)

        # ۲. استنتاج مدل زبانی
        t_gen_start = time.perf_counter()
        answer = generate_answer(query, contexts)
        generation_ms = round((time.perf_counter() - t_gen_start) * 1000, 2)
        
        # ۳. ارزیابی خودکار با RAG Triad
        evaluation = RagEvaluator.evaluate_rag_triad(query=query, answer=answer, contexts=contexts)
        total_latency_ms = round((time.perf_counter() - total_start) * 1000, 2)

    # نمایش پاسخ و شواهد
    col_ans, col_chunks = st.columns([6, 4])
    with col_ans:
        st.subheader("💡 پاسخ مستند سیستم:")
        st.info(answer)

    with col_chunks:
        st.subheader("📑 شواهد استخراج‌شده (Retrieved Clauses):")
        if contexts:
            for idx, c in enumerate(contexts):
                score = c.get("rerank_score", 0.0)
                clause_id = c.get("metadata", {}).get("clause_id", "نامشخص")
                with st.expander(f"مستند {idx+1}: {c['source']} (صفحه {c['page']})"):
                    st.write(c["text"])
                    st.caption(f"شناسه بند: `{clause_id}` | امتیاز بازچینی (Rerank): `{score:.4f}`")
        else:
            st.warning("هیچ قطعه مرتبطی یافت نشد.")

    st.divider()

    # رصدپذیری سازمانی (Observability & Metrics Dashboard)
    st.subheader("📊 داشبورد رصدپذیری و سنجش کیفیت (Observability & Evaluation)")
    
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    
    with m_col1:
        st.metric(
            label="⏱ زمان کل (Total Latency)",
            value=f"{total_latency_ms} ms"
        )
    with m_col2:
        st.metric(
            label="🔍 زمان بازیابی (Retrieval)",
            value=f"{retrieval_ms} ms"
        )
    with m_col3:
        st.metric(
            label="🧠 زمان تولید مدل (LLM)",
            value=f"{generation_ms} ms"
        )
    with m_col4:
        st.metric(
            label="🎯 ارتباط کانتکست (Relevance)",
            value=f"{int(evaluation['context_relevance'] * 100)}%"
        )
    with m_col5:
        faith_val = int(evaluation["faithfulness"] * 100)
        st.metric(
            label="🛡 وفاداری به متن (Faithfulness)",
            value=f"{faith_val}%"
        )