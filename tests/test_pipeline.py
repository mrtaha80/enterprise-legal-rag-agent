import pytest
from unittest.mock import patch, MagicMock
from src.parser import extract_chunks_from_pdf
from src.generator import generate_answer

class TestLegalRagPipeline:

    def test_01_ingestion_and_metadata(self, sample_pdf_path):
        """تست استخراج متنی و صحت ابرداده‌های صفحه‌بندی"""
        chunks = extract_chunks_from_pdf(sample_pdf_path)
        
        assert len(chunks) >= 2, "تعداد قطعات استخراج‌شده نباید کمتر از صفحات باشد."
        pages = {c["metadata"]["page"] for c in chunks}
        assert 1 in pages and 2 in pages, "ابرداده‌ی صفحات باید شامل صفحه ۱ و ۲ باشد."
        assert all("content" in c and len(c["content"].strip()) > 0 for c in chunks), "محتوای چانک‌ها نباید خالی باشد."

    def test_02_indexing_and_retrieval(self, sample_pdf_path, vector_db, retriever):
        """تست ذخیره‌سازی در Qdrant و بازیابی بر اساس شباهت معنایی"""
        chunks = extract_chunks_from_pdf(sample_pdf_path)
        vector_db.insert_chunks(chunks)

        query = "What is the penalty for late delivery?"
        results = retriever.retrieve(query=query, top_k=2)

        assert len(results) > 0, "موتور بازیابی باید حداقل یک قطعه مرتبط بازگرداند."
        top_match = results[0]
        assert "1,000 EUR" in top_match["text"], "چانک اول باید مستقیماً به جریمه ۱۰۰۰ یورویی اشاره کند."
        assert top_match["page"] == 2, "بند جریمه باید مشخصاً به صفحه ۲ ارجاع داده شده باشد."
        assert "rerank_score" in top_match, "نمره Re-ranking باید در متادیتا ثبت شده باشد."

    def test_03_citation_extraction(self, retriever):
        """تست حفظ یکپارچگی استناد به سند و صفحه در خروجی ریتریور"""
        results = retriever.retrieve(query="Total contract value and phases", top_k=1)
        assert len(results) == 1
        assert results[0]["page"] == 1, "ارزش قرارداد باید از صفحه ۱ بازیابی شود."
        assert "source" in results[0] and results[0]["source"].endswith(".pdf")

    @patch("src.generator.client.chat.completions.create")
    def test_04_e2e_answer_and_grounding(self, mock_create, retriever):
        """تست تولید پاسخ نهایی توسط مدل با استناد دقیق"""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="The liquidated damages penalty is 1,000 EUR per business day (test_contract.pdf, Page 2)."))
        ]
        mock_create.return_value = mock_response

        query = "What is the liquidated damage per day?"
        contexts = retriever.retrieve(query=query, top_k=2)
        answer = generate_answer(query=query, retrieved_contexts=contexts)

        assert answer is not None and len(answer) > 0
        assert ("1,000" in answer or "1000" in answer), "پاسخ مدل باید مبلغ دقیق جریمه را شامل شود."

    @patch("src.generator.client.chat.completions.create")
    def test_05_anti_hallucination_guardrail(self, mock_create, retriever):
        """تست گاردریل ضد-توهم برای موضوعات خارج از متن قرارداد"""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="با توجه به اسناد بارگذاری‌شده، پاسخی برای این سوال یافت نشد."))
        ]
        mock_create.return_value = mock_response

        unrelated_query = "What are the rules regarding annual employee health insurance?"
        contexts = retriever.retrieve(query=unrelated_query, top_k=2)
        answer = generate_answer(query=unrelated_query, retrieved_contexts=contexts)

        refusal_markers = [
            "یافت نشد",
            "اشاره‌ای نشده",
            "ذکر نشده",
            "پاسخی برای این سوال یافت نشد",
            "no mention",
            "not found"
        ]
        assert any(marker in answer.lower() for marker in refusal_markers), (
            f"مدل نباید به اطلاعات غایب پاسخ مثبت دهد. پاسخ دریافت‌شده: {answer}"
        )

    def test_06_clause_aware_chunking_integrity(self, sample_pdf_path):
        """تست یکپارچگی مواد حقوقی و وجود متادیتای اختصاصی بند"""
        chunks = extract_chunks_from_pdf(sample_pdf_path)
        
        # ۱. بررسی وجود متادیتای ساختاریافته بند (clause_id) در تمام چانک‌ها
        assert all("clause_id" in c["metadata"] for c in chunks), "هر چانک باید دارای شناسه ساختاریافته بند (clause_id) باشد."
        
        # ۲. بررسی تفکیک‌ناپذیری متن بند و عنوان ماده حقوقی
        penalty_chunk = next((c for c in chunks if "1,000 EUR" in c["content"]), None)
        assert penalty_chunk is not None, "بند جریمه باید استخراج شده باشد."
        assert "Article 2" in penalty_chunk["content"], "عنوان ماده (Article 2) نباید از مفاد جریمه جدا شده باشد."