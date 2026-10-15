import os
import requests
from src.logger import pipeline_logger, logger

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")
MODEL_NAME = os.getenv("LLM_MODEL", "llama3.2")

def generate_answer(query: str, contexts: list) -> str:
    if not contexts:
        return "در متن ارائه‌شده اشاره‌ای به این موضوع نشده است."

    # خلاصه‌سازی و آماده‌سازی متن چانک‌ها جهت کاهش حجم توکن ورودی
    context_blocks = []
    for idx, c in enumerate(contexts, 1):
        text = (c.get("text") or c.get("content") or "").strip()
        source = c.get("source", "مستند")
        page = c.get("page", 1)
        context_blocks.append(f"[{idx}] سند: {source} (صفحه {page}):\n{text}")

    combined_context = "\n\n".join(context_blocks)

    system_instruction = (
        "شما یک دستیار هوشمند استخراج و تحلیل حقوقی قراردادها هستید. "
        "با تکیه دقیق بر متن مستندات زیر، به سوال کاربر به زبان فارسی پاسخ شفاف و دقیق دهید. "
        "در صورتی که پاسخ شامل ارقام مالی، مبالغ جریمه، مهلت‌ها یا مواعد تحویل است، آن را دقیق ذکر کنید و شماره سند یا صفحه را ارجاع دهید. "
        "تنها در صورتی که واقعاً هیچ اطلاعاتی در متن وجود ندارد بگویید: در متن ارائه‌شده اشاره‌ای به این موضوع نشده است."
    )

    prompt = f"""{system_instruction}

متن مستندات قرارداد:
{combined_context}

پرسش کاربر:
{query}

پاسخ حقوقی مستند:"""

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
                        "num_ctx": 3072
                    }
                },
                timeout=120  # مهلت کافی برای اجرای پاسخ روی پردازنده
            )
            if response.status_code == 200:
                data = response.json()
                ans = data.get("response", "").strip()
                return ans if ans else "در متن ارائه‌شده اشاره‌ای به این موضوع نشده است."
            
            logger.error(f"Ollama returned status code: {response.status_code}")
            return "در متن ارائه‌شده اشاره‌ای به این موضوع نشده است."

    except Exception as e:
        logger.error(f"Error during LLM generation: {str(e)}")
        return "در متن ارائه‌شده اشاره‌ای به این موضوع نشده است."