import sys
from pathlib import Path

# اضافه کردن ریشه پروژه به مسیر ماژول‌های پایتون
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import os
import fitz
from src.vector_store import VectorDBService
from src.retriever import HybridRetriever
# ادامه کدهای قبلی...

import pytest
import os
import fitz
from src.vector_store import VectorDBService
from src.retriever import HybridRetriever

@pytest.fixture(scope="session")
def sample_pdf_path(tmp_path_factory):
    """ایجاد یک فایل PDF چندصفحه‌ای برای محیط تست ایزوله"""
    pdf_dir = tmp_path_factory.mktemp("contracts")
    pdf_path = str(pdf_dir / "test_contract.pdf")

    doc = fitz.open()

    # صفحه ۱: بند مالی و مشخصات
    p1 = doc.new_page()
    p1.insert_text(fitz.Point(50, 70), "Contract Agreement - Testing Module", fontsize=14)
    p1.insert_text(fitz.Point(50, 110), "Article 1: The Total Contract Value is 100,000 EUR.", fontsize=11)
    p1.insert_text(fitz.Point(50, 140), "Payment will be executed in 4 equal phases.", fontsize=11)

    # صفحه ۲: جریمه‌ها و فسخ
    p2 = doc.new_page()
    p2.insert_text(fitz.Point(50, 70), "Article 2: Liquidated Damages and Termination", fontsize=14)
    p2.insert_text(fitz.Point(50, 110), "Late delivery penalty is fixed at 1,000 EUR per business day.", fontsize=11)
    p2.insert_text(fitz.Point(50, 140), "Termination notice requires 30 days prior written confirmation.", fontsize=11)

    doc.save(pdf_path)
    doc.close()
    return pdf_path

@pytest.fixture(scope="session")
def vector_db():
    return VectorDBService()

@pytest.fixture(scope="session")
def retriever():
    return HybridRetriever()