import os
import requests
from src.logger import pipeline_logger, logger

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")
MODEL_NAME = os.getenv("LLM_MODEL", "llama3.2")

def generate_answer(query: str, contexts: list) -> str:
    if not contexts:
        return "در متن ارائه‌شده اشاره‌ای به این موضوع نشده است."

    context_blocks = []
    for idx, c in enumerate(contexts, 1):
        text = (c.get("text") or c.get("content") or "").strip()
        source = c.get("source", "contract.pdf")
        page = c.get("page", 1)
        if text:
            context_blocks.append(f"--- Document [{idx}]: {source} (Page {page}) ---\n{text}")

    if not context_blocks:
        return "در متن ارائه‌شده اشاره‌ای به این موضوع نشده است."

    combined_context = "\n\n".join(context_blocks)

    prompt = f"""You are a precise enterprise legal and financial AI assistant.
Your task is to answer the user's Persian question accurately based ONLY on the provided contract clauses below.
Note: The contract clauses may be in English or Persian. Translate and map legal concepts accurately (for example: Penalty / Liquidated Damages / Delay = جریمه تاخیر در تحویل, Contract Value / Fee = مبلغ قرارداد, Termination = فسخ).

Rules:
1. Write your final answer in clear, professional Persian (فارسی).
2. Extract exact numbers, percentages, currencies (EUR, USD, etc.), time limits, and cite the Document name and Page number.
3. ONLY if the provided clauses contain zero relevant information about the topic, reply exactly:
"در متن ارائه‌شده اشاره‌ای به این موضوع نشده است."

Contract Clauses:
{combined_context}

User Question (in Persian):
{query}

Answer in Persian (پاسخ مستند به فارسی):"""

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
                        "num_predict": 250,
                        "num_ctx": 2048
                    }
                },
                timeout=120
            )
            if response.status_code == 200:
                data = response.json()
                ans = data.get("response", "").strip()
                if ans:
                    return ans
                return "در متن ارائه‌شده اشاره‌ای به این موضوع نشده است."
            
            logger.error(f"Ollama HTTP {response.status_code}: {response.text}")
            return f"⚠️ خطا در سرویس مدل زبانی (کد {response.status_code}): لطفاً مطمئن شوید مدل {MODEL_NAME} در Ollama دانلود شده است."

    except Exception as e:
        logger.error(f"Error during LLM generation: {str(e)}")
        return f"⚠️ خطا در ارتباط با موتور Ollama: {str(e)}"