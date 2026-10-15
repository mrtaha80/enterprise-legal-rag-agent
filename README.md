# ⚖️ Enterprise Legal & Financial RAG Agent

[![CI Pipeline](https://github.com/mrtaha80/enterprise-legal-rag-agent/actions/workflows/test.yml/badge.svg)](https://github.com/mrtaha80/enterprise-legal-rag-agent/actions)
![Python Version](<https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg?logo=python>)
![Vector Database](https://img.shields.io/badge/Vector_DB-Qdrant-DC2626.svg?logo=qdrant)
![UI Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg?logo=streamlit)
![Evaluation](https://img.shields.io/badge/Evaluation-RAG_Triad-10B981.svg)
![License](https://img.shields.io/badge/License-MIT-gray.svg)

سامانه سازمانی پرسش‌وپاسخ، تحلیل معنایی و اعتبارسنجی قراردادهای حقوقی و اسناد مالی تجاری مبتنی بر متدولوژی تفکیک بندمحور (Clause-Aware)، رتبه‌بندی مجدد و سپر ضد-توهم.

---

## 📌 معماری سیستم (System Architecture)

```text
[اسناد حقوقی (PDF)]
        │
        ▼
[Clause-Aware Parser] ──(تفکیک مواد قانونی و ابرداده صفحه)──┐
        │                                                  │
        ▼                                                  ▼
[Embeddings]                                     [شناسه بند و ابرداده]
        │                                                  │
        ▼                                                  │
[Qdrant Vector Database] ◄─────────────────────────────────┘
        │
        ├─── [پرسش کاربر]
        ▼
[Hybrid Retrieval (Top-K)]
        │
        ▼
[Cross-Encoder Reranker]
        │
        ▼
[Anti-Hallucination Guardrail]
        │
        ▼
[LLM Engine (Llama 3.2)] ──► [پاسخ مستند با شماره صفحه و سند]
        │
        ▼
[RAG Triad Evaluator & Latency Tracer] ──► [داشبورد Streamlit & لاگ‌های JSON]
```

---

## ✨ قابلیت‌های کلیدی

* **خردایش بندمحور (Clause-Aware Chunking):** تفکیک متون بر اساس ساختار مواد و تبصره‌های قانونی به جای قطعه‌بندی کاراکتری کور.
* **بازیابی دوسطحی (Hybrid Search & Cross-Encoder Reranking):** بهینه‌سازی کاندیداهای متنی با بازچینی عمیق معنایی.
* **سپر ضد-توهم (Anti-Hallucination Guardrail):** امتناع هوشمند از پاسخ‌دهی در صورت عدم حضور شواهد در مستندات.
* **ارزیابی کیفی RAG Triad:** اندازه‌گیری دقیق شاخص‌های وفاداری به متن (Faithfulness) و ربط کانتکست (Context Relevance).
* **رصدپذیری سازمانی (Observability):** پایش میلی‌ثانیه‌ای تاخیر لایه‌ها و ثبت ساختاریافته رویدادها در قالب JSON.
* **تست‌های خودکار و CI/CD:** پوشش کامل پایپ‌لاین با ۸ تست در Pytest و اجرای خودکار در GitHub Actions با کانتینر Qdrant.

---

## 📊 نتایج ارزیابی و بنچمارک

| شاخص ارزیابی                                    |    مقدار ثبت‌شده    | وضعیت استاندارد |
| :--------------------------------------------------------- | :-----------------------------: | :---------------------------: |
| **وفاداری به متن (Faithfulness)**        |        **۹۲٪**        | پاس‌شده ($\ge 85\%$) |
| **ارتباط کانتکست (Context Relevance)**  |        **۷۸٪**        | پاس‌شده ($\ge 70\%$) |
| **دقت استناد (Citation Accuracy)**          |       **۱۰۰٪**       |        پاس‌شده        |
| **تست‌های پایپ‌لاین (Pytest Suite)** | **۸ از ۸ (۱۰۰٪)** |   آماده استقرار   |

---

## 💡 Example Usage: Query to Grounded Legal Answer

| Phase                      | Output / Component                                                                                              |
| :------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| **User Query**       | "شرایط فسخ قرارداد اجاره به دلیل عدم پرداخت اجاره‌بها چیست؟"   |
| **Hybrid Retrieval** | `commercial_lease_agreement_1402.pdf` (Page 4, Clause 12-3)                                                   |
| **Grounded Answer**  | موجر پس از ۱۵ روز تاخیر و ارسال اخطار کتبی ۱۰ روزه حق فسخ دارد. |
| **Evaluation**       | Faithfulness: 0.98 \| Context Relevance: 0.94                                                                   |





🚀 راهنمای نصب و اجرا

### ۱. راه‌اندازی پایگاه داده برداری Qdrant

```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

### ۲. نصب پیش‌نیازها

```bash
pip install -r requirements.txt
```

### ۳. اجرای تست‌های خودکار

```bash
python -m pytest -v -s
```

### ۴. اجرای برنامه

```bash
streamlit run app.py
```
