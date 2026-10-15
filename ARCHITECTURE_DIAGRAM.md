# 🏛️ Enterprise Legal RAG Agent — Comprehensive System Architecture

This document illustrates the end-to-end architecture of the Enterprise Legal RAG Agent, highlighting data flow, security guardrails, hybrid retrieval mechanics, and evaluation loops.

---

## 📊 End-to-End System Architecture Diagram

```mermaid
graph TD
    %% Styling
    classDef client fill:#f9f,stroke:#333,stroke-width:2px;
    classDef security fill:#ff9999,stroke:#333,stroke-width:2px;
    classDef pipeline fill:#99ccff,stroke:#333,stroke-width:2px;
    classDef retrieval fill:#99ff99,stroke:#333,stroke-width:2px;
    classDef ai fill:#ffcc99,stroke:#333,stroke-width:2px;
    classDef storage fill:#d9d9d9,stroke:#333,stroke-width:2px;

    %% Client Layer
    User([👤 User / Legal Expert]) -->|Query / PDF Upload| UI[🖥️ Streamlit Dashboard UI]:::client

    %% Security & Ingestion Layer
    UI -->|Raw PDF / Query| SecGuard{🛡️ Security Guard}:::security
    SecGuard -->|Prompt Injection Check & PII Masking| Parser[📄 PDF Parser & Clause Splitter]:::security
    Parser -->|Cleaned Document Chunks| VectorMgr[(🔄 Qdrant Vector Manager)]:::storage

    %% Pipeline Orchestration
    UI -->|Search Query| Pipeline[⚙️ LegalRAGPipeline Service]:::pipeline

    %% Retrieval & Re-ranking Subsystem
    Pipeline --> Retriever[🔍 HybridRetriever]:::retrieval
    Retriever -->|Bilingual Concept Expansion| Qdrant[(🗄️ Qdrant Vector DB<br/>Cosine Similarity)]:::storage
    Qdrant -->|Top-K Dense Candidates| CrossEncoder[⚡ Neural Cross-Encoder<br/>mmarco-mMiniLMv2]:::retrieval
    CrossEncoder -->|Hybrid Score Fusion & Reranking| Contexts[📋 Ranked Legal Contexts]:::retrieval

    %% Generation & Evaluation Subsystem
    Contexts --> Generator[🤖 Llama 3.2 Generator via Ollama]:::ai
    Generator -->|Grounded Legal Answer + Citations| Evaluator[📊 RagEvaluator (RAG Triad)]:::ai
    
    Evaluator -->|Faithfulness & Context Relevance Score| UI
    Generator -->|Final Secure Answer & Citations| UI

    %% Subgraph Grouping
    subgraph Docker Container Stack (WSL 2 / Linux)
        VectorMgr
        Qdrant
        Generator
    end
