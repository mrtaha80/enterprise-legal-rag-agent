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
Answer the user's Persian question accurately based ONLY on the provided contract clauses below.
Map English legal terms to Persian accurately (e.g., Penalty / Liquidated Damages / Delay = جریمه تاخیر در تحویل, Contract Value / Fee = مبلغ قرارداد, Termination = فسخ).

Rules:
1. Write your final answer in clear Persian (فارسی).
2. Extract exact numbers, percentages, currencies (EUR, USD, etc.), time limits, and cite the Document name and Page number.
3. ONLY if the clauses contain zero relevant information, reply exactly:
"در متن ارائه‌شده اشاره‌ای به این موضوع نشده است."

Contract Clauses:
{combined_context}

User Question:
{query}

Answer in Persian:"""

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
                        "num_predict": 200,
                        "num_ctx": 1024  # بهینه‌سازی مصرف رم برای جلوگیری از خطای 500 روی رم 8GB
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
            
            # نمایش دقیق دلیل خطای داخلی Ollama در صورت رخداد
            err_detail = response.text
            logger.error(f"Ollama HTTP {response.status_code}: {err_detail}")
            return f"⚠️ خطا در سرویس مدل زبانی (کد {response.status_code}): {err_detail}"

    except Exception as e:
        logger.error(f"Error during LLM generation: {str(e)}")