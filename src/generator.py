import os
import requests
from src.logger import pipeline_logger

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("LLM_MODEL", "llama3.2")

def generate_answer(query: str, contexts: list) -> str:
    combined_context = "\n\n".join([f"Document [{c.get('source', '')} - Page {c.get('page', '')}]:\n{c.get('text', '')}" for c in contexts])
    
    prompt = f"""You are a professional legal AI assistant. Answer the user question based strictly on the provided context below.
If the information is not mentioned in the context, explicitly state: "در متن ارائه‌شده اشاره‌ای به این موضوع نشده است."

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
                        "num_predict": 200  # جلوگیری از لوپ و تولید بی‌پایان
                    }
                },
                timeout=45  # حداکثر ۴۵ ثانیه منتظر بماند و گیر نکند
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "").strip()
            else:
                return "خطا در ارتباط با مدل زبانی لوکال."
    except requests.exceptions.Timeout:
        return "در متن ارائه‌شده اشاره‌ای به این موضوع نشده است (زمان پاسخ‌دهی به پایان رسید)."
    except Exception as e:
        return f"خطا در فراخوانی مدل: {str(e)}"