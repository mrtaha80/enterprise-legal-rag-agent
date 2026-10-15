
# 🏛️ Enterprise Legal RAG Agent — Comprehensive System Architecture

This document illustrates the end-to-end architecture of the Enterprise Legal RAG Agent, highlighting data flow, security guardrails, hybrid retrieval mechanics, and evaluation loops.

---

## 📊 End-to-End System Architecture Diagram

```mermaid
graph TD
    %% Modern Enterprise Color Palette & Styling
    classDef client fill:#e0e7ff,stroke:#3730a3,stroke-width:2px,color:#1e1b4b,font-weight:bold;
    classDef security fill:#ffe4e6,stroke:#9f1239,stroke-width:2px,color:#881337,font-weight:bold;
    classDef pipeline fill:#ccfbf1,stroke:#0f766e,stroke-width:2px,color:#134e4a,font-weight:bold;
    classDef retrieval fill:#d1fae5,stroke:#047857,stroke-width:2px,color:#064e3b,font-weight:bold;
    classDef ai fill:#fef3c7,stroke:#b45309,stroke-width:2px,color:#78350f,font-weight:bold;
    classDef storage fill:#f4f4f5,stroke:#3f3f46,stroke-width:2px,color:#18181b,font-weight:bold;

    %% Client Layer
    User([User / Legal Expert]) -->|Query / PDF Upload| UI[Streamlit Dashboard UI]:::client

    %% Security & Ingestion Layer
    UI -->|Raw PDF / Query| SecGuard{Security Guard}:::security
    SecGuard -->|Prompt Injection Check & PII Masking| Parser[PDF Parser & Clause Splitter]:::security
    Parser -->|Cleaned Chunks| VectorMgr[(Qdrant Vector Manager)]:::storage

    %% Pipeline Orchestration
    UI -->|Search Query| Pipeline[LegalRAGPipeline Service]:::pipeline

    %% Retrieval & Re-ranking Subsystem
    Pipeline --> Retriever[HybridRetriever]:::retrieval
    Retriever -->|Bilingual Concept Expansion| Qdrant[(Qdrant Vector DB)]:::storage
    Qdrant -->|Top-K Dense Candidates| CrossEncoder[Neural Cross-Encoder]:::retrieval
    CrossEncoder -->|Hybrid Score Fusion| Contexts[Ranked Legal Contexts]:::retrieval

    %% Generation & Evaluation Subsystem
    Contexts --> Generator[Llama 3.2 Generator via Ollama]:::ai
    Generator -->|Grounded Answer & Citations| Evaluator[RagEvaluator Module]:::ai
  
    Evaluator -->|Faithfulness & Relevance Score| UI
    Generator -->|Final Secure Answer| UI
```
