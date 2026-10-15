# 🏛️ Enterprise Legal RAG Agent — Architecture

```mermaid
graph TD
    Client --> UI
    UI --> Guard
    Guard --> Parser
    Parser --> Qdrant
    UI --> Pipeline
    Pipeline --> Retriever
    Retriever --> Qdrant
    Qdrant --> Reranker
    Reranker --> Generator
    Generator --> Evaluator
    Evaluator --> UI
```
