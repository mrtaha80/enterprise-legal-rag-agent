import streamlit as st
import tempfile
import os
from src.parser import extract_chunks_from_pdf
from src.vector_store import VectorDBService
from src.retriever import HybridRetriever
from src.generator import generate_answer

st.set_page_config(page_title="Legal & Financial RAG Agent", layout="wide")

@st.cache_resource
def get_services():
    db = VectorDBService()
    retriever = HybridRetriever()
    return db, retriever

db_service, retriever = get_services()

st.title("📄 سامانه هوشمند تحلیل قراردادها و اسناد (RAG)")

with st.sidebar:
    st.header("مدیریت اسناد")
    uploaded_files = st.file_uploader("فایل‌های PDF را بارگذاری کنید", type=["pdf"], accept_multiple_files=True)
    if st.button("پردازش و ذخیره در دیتابیس"):
        if uploaded_files:
            total_chunks = 0
            with st.spinner("در حال پردازش و استخراج بردارها..."):
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    
                    chunks = extract_chunks_from_pdf(tmp_path)
                    for c in chunks:
                        c["metadata"]["source"] = uploaded_file.name
                    db_service.insert_chunks(chunks)
                    total_chunks += len(chunks)
                    os.remove(tmp_path)
            st.success(f"{total_chunks} چانک با موفقیت ایندکس شد.")
        else:
            st.warning("ابتدا یک فایل انتخاب کنید.")

query = st.text_input("پرسش خود را درباره مفاد قرارداد یا سند بنویسید:")
if st.button("جستجو و تحلیل") and query:
    with st.spinner("بازیابی و بازچینی مستندات..."):
        contexts = retriever.retrieve(query, top_k=3)
        answer = generate_answer(query, contexts)

    col1, col2 = st.columns([6, 4])
    with col1:
        st.subheader("پاسخ مستند مدل:")
        st.write(answer)
    with col2:
        st.subheader("شواهد استخراج‌شده (Retrieved Chunks):")
        for idx, c in enumerate(contexts):
            with st.expander(f"منبع {idx+1}: {c['source']} (صفحه {c['page']})"):
                st.write(c["text"])
                if "rerank_score" in c:
                    st.caption(f"امتیاز بازچینی (Rerank Score): {c['rerank_score']:.4f}")