# ⚖️ Enterprise Legal RAG System — Advanced Retrieval Architecture

This document provides a rigorous technical breakdown of the **Two-Stage Hybrid Retrieval and Neural Re-ranking Pipeline** implemented in the Enterprise Legal RAG system. Designed specifically for complex legal and financial corpora, this architecture bridges the semantic, multilingual, and precision gaps inherent in standard vector search implementations.

---

## 🏗️ Architecture Overview

The retrieval process operates through a sophisticated sequential pipeline ensuring both high recall (not missing relevant clauses) and high precision (ranking exact legal obligations, penalties, and time limits at the top).

[ User Query (Persian) ]
│
▼
[ Bilingual Query Expansion ] ──> (Enriches with English legal equivalents: Penalty, Delay, etc.)
│
▼
[ Stage 1: Dense Vector Retrieval (Qdrant) ] ──> (Cosine Similarity via multilingual-e5-base)
│
▼
[ Stage 2: Neural Cross-Encoder Re-ranking ] ──> (Deep cross-attention via mmarco-mMiniLMv2)
│
▼
[ Hybrid Score Fusion & Top-K Filtering ] ──> (Passed to Llama 3.2 Generator)


---

## 🔍 Detailed Pipeline Stages

### 1. Bilingual Query Expansion (`src/retriever.py`)
* **The Challenge:** Legal contracts (especially in enterprise environments) frequently contain mixed-language terminology or English clauses (e.g., *Liquidated Damages*, *Force Majeure*), whereas user queries are typically formulated in Persian.
* **The Solution:** The retriever automatically inspects the query against a legal concept map (`RagEvaluator.CONCEPT_MAP`), dynamically appending precise English legal equivalents to the search query to eliminate cross-lingual semantic gaps.

### 2. Stage 1 — Dense Vector Retrieval (`src/vector_store.py`)
* **Embedding Model:** `intfloat/multilingual-e5-base` (State-of-the-art dense encoder optimized for multilingual representation).
* **Vector Database:** **Qdrant** configured with **Cosine Similarity** metric.
* **Mechanism:** Converts expanded queries into dense vectors and performs fast approximate nearest neighbor (ANN) search across clause-aware chunked documents, retrieving top candidate passages.

### 3. Stage 2 — Neural Cross-Encoder Re-ranking (`src/retriever.py`)
* **Re-ranking Model:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
* **Mechanism:** Unlike bi-encoders (which encode queries and documents independently), the cross-encoder processes the query and each candidate chunk *simultaneously* via full cross-attention layers, capturing fine-grained contextual nuances.
* **Hybrid Score Fusion:** Combines dense vector similarity scores with normalized cross-encoder probabilities using a balanced weighting formula:
  $$\text{Score}_{\text{hybrid}} = (0.6 \times S_{\text{vector}}) + (0.4 \times S_{\text{rerank}})$$

---

## 📊 Observability & RAG Triad Evaluation

To ensure enterprise-grade reliability, every retrieval and generation cycle is evaluated in real time across the **RAG Triad** metrics integrated directly into the Streamlit dashboard (`src/evaluator.py`):
1. **Context Relevance:** Measures how accurately the retrieved clauses cover the concepts requested in the user query.
2. **Faithfulness:** Validates that the generated LLM response is strictly grounded in the retrieved context, preventing hallucinations.

---
*Maintained by Engineering Team • Enterprise Legal RAG Agent*
