
# ⚖️ Persian-First Legal & Financial RAG System

[![CI Pipeline](https://github.com/mrtaha80/enterprise-legal-rag-agent/actions/workflows/test.yml/badge.svg)](https://github.com/mrtaha80/enterprise-legal-rag-agent/actions)
![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python)
![Vector Database](https://img.shields.io/badge/Vector_DB-Qdrant-DC2626.svg?logo=qdrant)
![UI Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-gray.svg)

A **Persian-first legal RAG system** featuring clause-aware PDF ingestion, hybrid dense-lexical retrieval with RRF fusion, multi-tenant data isolation, deterministic metadata citations, executable end-to-end benchmarking, and security-aware document processing.

---

## 📌 معماری سیستم (System Architecture)

```text
[اسناد حقوقی (PDF)]
        │
        ▼
[Clause-Aware Parser (Tail-Safe Sliding Window)]
        │
        ▼
[SecurityGuard (Zero-Tolerance Prompt Injection Drop + PII Masking)]
        │
        ▼
[E5 Embeddings (passage: / query: prefixes) + Deterministic UUIDv5]
        │
        ▼
[Qdrant Vector Database (Tenant-Isolated Payload Indexes)]
        │
        ├───► [Dense Vector Search (Top-K)] ─────────────────────┐
        │                                                        ▼
        └───► [Full Paginated Scroll + Normalized Lexical] ──► [RRF Fusion]
                                                                 │
                                                                 ▼
                                                      [Cross-Encoder Reranker]
                                                                 │
                                                                 ▼
                                               [Data-Sandboxed LLM Engine (Llama 3.2)]
                                                                 │
                                                                 ▼
                                         [Deterministic Citations + XSS-Escaped UI]
```

---

## ✨ ویژگی‌های فنی و مهندسی

- **خردایش بندمحور ایمن (Tail-Safe Clause-Aware Chunking):** تفکیک اسناد بر اساس ساختار مواد، بندها و تبصره‌های قانونی با مرتب‌سازی هندسی بلوک‌ها (`sort=True`) و پنجره لغزان کلمه‌ای بدون از دست رفتن انتهای بندها.
- **بازیابی ترکیبی (Dense + Lightweight Normalized Lexical Overlap + RRF):** ترکیب جستجوی برداری (با پیشوندهای استاندارد `query:` و `passage:` در مدل `multilingual-e5-base`) و تطابق واژگانی نرمال‌شده فارسی (با پیمایش صفحه‌بندی‌شده کامل رکوردهای Qdrant) از طریق الگوریتم Reciprocal Rank Fusion و بازچینی با Cross-Encoder.
- **استناددهی قطعی (Deterministic Citations):** استخراج شماره صفحه، نام سند و شناسه بند (`clause_id`) مستقیماً از متاداده‌ی تاییدشده پایگاه داده، بدون اتکا به متن تولیدی مدل زبانی.
- **ایزوله‌سازی داده در سطح پایگاه داده (Multi-Tenant Data Isolation):** تفکیک کامل بردارها و اسناد هر فضای کاری با فیلترهای سطح کوئری (`tenant_id`) و ایندکس‌های Payload در Qdrant.
- **امنیت لایه‌ای و لاگینگ بدون نشت داده:** حذف چانک‌های آلوده به Prompt Injection، ماسک‌سازی خودکار اطلاعات شخصی (PII) در متون و لاگ‌های JSON (`secure_log_payload`)، مخفی‌سازی خطاهای داخلی سرور از دید کاربر، و ایمن‌سازی خروجی HTML در برابر XSS (`html.escape`).
- **بنچمارک اجرایی ۱۱ سناریویی (Executable Golden Benchmark Suite):** ارزیابی خودکار End-to-End روی سناریوهای فارسی، انگلیسی، عددی، چندبندی، چندصفحه‌ای، تضاد الحاقیه، خارج از موضوع و تست امنیت با تفکیک زمان Retrieval و Generation و قابلیت خروجی JSON.

---

## 📊 ارزیابی و پایش (Observability & Benchmarking)

> **⚠️ نکته فنی درباره متریک‌های لحظه‌ای UI:** معیارهای نمایش‌داده‌شده در کارت‌های داشبورد (`Heuristic Faithfulness Indicator` و `Heuristic Relevance Indicator`) شاخص‌های تقریبی (Proxy Indicators) مبتنی بر هم‌پوشانی واژگانی و مفهومی برای پایش لحظه‌ای هستند و **جایگزین ارزیابی انسانی یا بنچمارک‌های استاندارد معنایی (مانند Ragas یا NLI)** نمی‌باشند. برای ارزیابی واقعی سیستم، از تب **Executable Golden Benchmark** استفاده کنید.

| بخش ارزیابی                      |             نوع سنجه             | توضیح فنی                                                                                        |
| :----------------------------------------- | :--------------------------------------: | :------------------------------------------------------------------------------------------------------- |
| **Golden Benchmark Suite**           | **۱۱ سناریوی End-to-End** | ارزیابی واقعی فارسی، انگلیسی، عددی، تضاد الحاقیه و امنیت |
| **Citation Extraction**              |    **Deterministic (۱۰۰٪)**    | استخراج قطعی از متادادهٔ Qdrant بدون توهم مدل                            |
| **Heuristic Relevance Indicator**    |     **Lexical/Concept Proxy**     | شاخص کمکی هم‌پوشانی مفاهیم پرسش و اسناد بازیابی‌شده          |
| **Heuristic Faithfulness Indicator** |     **Lexical/Concept Proxy**     | شاخص کمکی هم‌پوشانی واژگان کلیدی پاسخ با متن مرجع               |

---

## 🛡️ شفاف‌سازی معماری و محدودیت‌های شناخته‌شده (Architectural Transparency & Limitations)

1. **جداسازی داده در برابر احراز هویت نمایشی (Tenant Isolation vs. Demo Auth):**
   > *Tenant isolation is implemented at the data layer, but the current identity gate is a demo shared-PIN mechanism and does not provide real user authentication.*
   > در محیط عملیاتی واقعی، ورودی نام کاربری و `APP_ACCESS_PIN` باید با یک سرویس احراز هویت استاندارد نظیر OAuth2 / OIDC یا JWT جایگزین شود.
   >
2. **ماهیت جستجوی واژگانی (Lightweight Lexical Overlap + RRF):**
   بخش واژگانی فعلی مبتنی بر نرمال‌سازی حروف فارسی («ی/ک»، نیم‌فاصله، ارقام) و هم‌پوشانی توکن‌ها بر روی تمام رکوردهای صفحه‌بندی‌شدهٔ کاربر در Qdrant است و با الگوریتم RRF با نتایج برداری ترکیب می‌شود؛ این مکانیزم جایگزین یک موتور کامل BM25 مبتنی بر Inverted Index مستقل نیست.
3. **انتخاب مدل بازچینی (Cross-Encoder Reranker):**
   مدل `mmarco-mMiniLMv2-L12-H384-v1` به عنوان یک Baseline چندزبانه سبک استفاده شده است. برای متون حقوقی سنگین فارسی، بنچمارک و جایگزینی با مدل‌های بزرگ‌تر نظیر `BAAI/bge-reranker-v2-m3` توصیه می‌شود.
4. **محدودیت اسناد اسکن‌شده (OCR):**
   پارسر فعلی برای فایل‌های PDF دارای لایه متنی دیجیتال بهینه‌سازی شده است و اسناد اسکن‌شدهٔ تصویری نیازمند اضافه شدن پیش‌پردازش OCR هستند.
5. **سلب مسئولیت حقوقی (Legal Disclaimer):**
   خروجی‌های این سامانه صرفاً جنبه کمکی و تحلیلی دارد و به هیچ عنوان جایگزین مشاوره حقوقی رسمی یا نظر وکیل دادگستری نمی‌باشد.

---

## 🚀 راهنمای نصب و اجرا

### ۱. تنظیم متغیرهای محیطی (`.env`)

```bash
cp .env.example .env
echo "APP_ACCESS_PIN=your_secure_demo_pin" >> .env
```

### ۲. راه‌اندازی پایگاه داده برداری Qdrant

```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

### ۳. نصب پیش‌نیازها و اجرای برنامه

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

*توسعه‌دهنده: طاها خدایاری*
