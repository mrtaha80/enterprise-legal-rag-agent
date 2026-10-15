import logging
from unittest.mock import patch
from src.parser import extract_chunks_from_pdf
from src.generator import generate_answer
from src.logger import trace_latency
from src.evaluator import RagEvaluator


class TestLegalRagPipeline:

    def test_01_ingestion_and_metadata(self, sample_pdf_path):
        """تست استخراج متنی و صحت ابرداده‌های صفحه‌بندی"""
        chunks = extract_chunks_from_pdf(sample_pdf_path)

        assert len(chunks) >= 2, "تعداد قطعات استخراج‌شده نباید کمتر از صفحات باشد."
        pages = {c["metadata"]["page"] for c in chunks}
        assert 1 in pages and 2 in pages, "ابرداده‌ی صفحات باید شامل صفحه ۱ و ۲ باشد."
        assert all("content" in c and len(c["content"].strip()) > 0 for c in chunks), "محتوای چانک‌ها نباید خالی باشد."

    def test_02_indexing_and_retrieval(self, sample_pdf_path, vector_db, retriever):
        """تست ذخیره‌سازی در Qdrant و بازیابی بر اساس شباهت معنایی و واژگانی"""
        chunks = extract_chunks_from_pdf(sample_pdf_path)
        vector_db.insert_chunks(chunks)

        query = "What is the penalty for late delivery?"
        results = retriever.retrieve(query=query, top_k=2)

        assert len(results) > 0, "موتور بازیابی باید حداقل یک قطعه مرتبط بازگرداند."
        top_match = results[0]
        text_body = top_match.get("text") or top_match.get("content", "")
        page_num = top_match.get("page") or top_match.get("metadata", {}).get("page")
        assert "1,000 EUR" in text_body, "چانک اول باید مستقیماً به جریمه ۱۰۰۰ یورویی اشاره کند."
        assert page_num == 2, "بند جریمه باید مشخصاً به صفحه ۲ ارجاع داده شده باشد."
        assert "rerank_score" in top_match, "نمره Re-ranking باید در خروجی ثبت شده باشد."

    def test_03_citation_extraction(self, retriever):
        """تست حفظ یکپارچگی استناد به سند و صفحه در خروجی ریتریور"""
        results = retriever.retrieve(query="Total contract value and phases", top_k=1)
        assert len(results) == 1
        page_num = results[0].get("page") or results[0].get("metadata", {}).get("page")
        source_name = results[0].get("source") or results[0].get("metadata", {}).get("source", "")
        assert page_num == 1, "ارزش قرارداد باید از صفحه ۱ بازیابی شود."
        assert source_name.endswith(".pdf")

    @patch("src.generator.LLMProviderClient.generate")
    def test_04_e2e_answer_and_grounding(self, mock_generate, retriever):
        """تست تولید پاسخ نهایی توسط مدل با API جدید (contexts=...)"""
        mock_generate.return_value = (
            "The liquidated damages penalty is 1,000 EUR per business day (test_contract.pdf, Page 2)."
        )

        query = "What is the liquidated damage per day?"
        contexts = retriever.retrieve(query=query, top_k=2)
        answer = generate_answer(query=query, contexts=contexts)

        assert answer is not None and len(answer) > 0
        assert ("1,000" in answer or "1000" in answer), "پاسخ مدل باید مبلغ دقیق جریمه را شامل شود."

    @patch("src.generator.LLMProviderClient.generate")
    def test_05_anti_hallucination_guardrail(self, mock_generate, retriever):
        """تست گاردریل ضد-توهم برای موضوعات خارج از متن قرارداد"""
        mock_generate.return_value = "در متن ارائه‌شده اشاره‌ای به این موضوع نشده است."

        unrelated_query = "What are the rules regarding annual employee health insurance?"
        contexts = retriever.retrieve(query=unrelated_query, top_k=2)
        answer = generate_answer(query=unrelated_query, contexts=contexts)

        refusal_markers = [
            "یافت نشد",
            "اشاره",
            "ذکر نشده",
            "پاسخی برای این سوال یافت نشد",
            "no mention",
            "not found",
        ]
        assert any(marker in answer.lower() for marker in refusal_markers), (
            f"مدل نباید به اطلاعات غایب پاسخ مثبت دهد. پاسخ دریافت‌شده: {answer}"
        )

    def test_06_clause_aware_chunking_integrity(self, sample_pdf_path):
        """تست یکپارچگی مواد حقوقی و وجود متادیتای اختصاصی بند"""
        chunks = extract_chunks_from_pdf(sample_pdf_path)

        assert all("clause_id" in c["metadata"] for c in chunks), (
            "هر چانک باید دارای شناسه ساختاریافته بند (clause_id) باشد."
        )

        penalty_chunk = next((c for c in chunks if "1,000 EUR" in c["content"]), None)
        assert penalty_chunk is not None, "بند جریمه باید استخراج شده باشد."
        assert "Article 2" in penalty_chunk["content"], "عنوان ماده (Article 2) نباید از مفاد جریمه جدا شده باشد."

    def test_07_structured_logging_and_latency(self, retriever, caplog):
        """تست تولید لاگ‌های ساختاریافته و ثبت تاخیر عملیات (Latency Tracing)"""
        with caplog.at_level(logging.INFO, logger="legal_rag"):
            with trace_latency("unit_test_operation", query="test query"):
                pass

        stage_logged = any(
            "unit_test_operation" in record.getMessage()
            or (hasattr(record, "extra_fields") and record.extra_fields.get("stage") == "unit_test_operation")
            for record in caplog.records
        )
        assert stage_logged, "مرحله عملیات و ترکینگ تاخیر باید با موفقیت در سیستم مانیتورینگ ثبت شود."

    def test_08_rag_triad_quality_metrics(self, retriever):
        """تست ارزیابی کیفی پاسخ و کانتکست بر اساس شاخص‌های وفاداری و ارتباط"""
        query = "What is the penalty for late delivery?"
        contexts = retriever.retrieve(query=query, top_k=2)
        answer = "The penalty for late delivery is 1,000 EUR per business day."

        evaluation = RagEvaluator.evaluate_rag_triad(query=query, answer=answer, contexts=contexts)

        assert evaluation["context_relevance"] >= 0.70, f"ارتباط کانتکست پایین است: {evaluation['context_relevance']}"
        assert evaluation["faithfulness"] >= 0.70, f"وفاداری پاسخ به مستندات پایین است: {evaluation['faithfulness']}"
        assert evaluation["overall_rag_score"] >= 0.70

    def test_09_strict_evaluator_rejects_false_refusal(self):
        """تست اطمینان از اینکه امتناع روی سوال داخل‌موضوع (is_out_of_scope=False) باعث شکست تست می‌شود"""
        in_scope_item = {
            "id": "TC-01",
            "category": "factual_penalty",
            "query": "جریمه تاخیر در تحویل فازها چقدر است؟",
            "expected_keywords": ["1,000", "یورو", "eur", "day"],
            "expected_page": 2,
            "is_out_of_scope": False,
        }
        contexts = [{"content": "Late delivery penalty is 1,000 EUR per day.", "page": 2}]
        false_refusal_answer = "در متن ارائه‌شده اشاره‌ای به این موضوع نشده است."

        result = RagEvaluator.evaluate_single_dataset_item(
            item=in_scope_item,
            answer=false_refusal_answer,
            contexts=contexts,
            ret_ms=120.0,
            gen_ms=450.0,
        )
        assert result["passed"] is False, "امتناع روی سوال داخل‌موضوع باید شکست تلقی شود!"
        assert result["outcome_type"] == "false_refusal"