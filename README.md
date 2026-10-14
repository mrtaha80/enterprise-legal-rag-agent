[![CI Pipeline](https://github.com/mrtaha80/enterprise-legal-rag-agent/actions/workflows/test.yml/badge.svg)](https://github.com/mrtaha80/enterprise-legal-rag-agent/actions)

# Enterprise Legal & Financial RAG Agent

A production-grade Retrieval-Augmented Generation (RAG) system engineered for high-precision contractual and financial document analysis with source attribution.

## Key Features

- **Hybrid Retrieval**: Dense vectors paired with lexical search.
- **Two-Stage Re-ranking**: Cross-encoder scoring to filter hallucinations.
- **Exact Metadata Lineage**: Page-level citation.
- **Vector Database**: Self-hosted Qdrant deployment via Docker.

## Tech Stack

- **Vector DB**: Qdrant
- **Framework**: Sentence-Transformers, PyMuPDF, Streamlit
- **LLM**: GPT-4o-mini / DeepSeek
- **Embedding**: intfloat/multilingual-e5-base
- **Re-ranker**: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
