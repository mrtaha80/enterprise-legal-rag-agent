import os
import requests
from src.logger import pipeline_logger

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("LLM_MODEL", "llama3.2")

def generate_answer(query: str, contexts: list) -> str:
    combined_context = "\n\n".join([
        f"Document [{c.get('source', '')} - Page {c.get('page', '')}]:\n{c.get('text', '')}"
        for c in contexts
    ])
    
    prompt = f"""You are an enterprise legal reasoning AI assistant. Answer the question accurately using the provided context below.
Notice: The context may be in English and the user query in Persian. Map key legal/financial terms semantically (e.g., penalty/delay = جریمه/تاخیر, contract value/payment = مبلغ قرارداد/پرداخت).

Rules:
1. If the information exists in the context, answer concisely and cite the document name and page number.
2. If and ONLY if the information is truly not mentioned in the context, explicitly reply: "در متن ارائه‌شده اشاره‌ای به این موضوع نشده است."

Context:
{combined_context}

Question:
{query}

Answer:"""

    try:
        with pipeline_logger.track_latency("llm_generation", query=query):
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 180
                    }
                },
                timeout=35
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "").strip()
            return "در متن ارائه‌شده اشاره‌ای به این موضوع نشده است."
    except Exception:
        return "در متن ارائه‌شده اشاره‌ای به این موضوع نشده است."