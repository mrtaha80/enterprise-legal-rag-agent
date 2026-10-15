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
Translate and extract exact numbers, penalties, currencies, or conditions directly into clear Persian.

Rules:
1. Provide ONLY the direct, final Persian answer. Do not repeat the question or add contradictory statements.
2. If the answer exists, state it clearly citing the rule or percentage.
3. If and only if the information is completely missing, reply exactly:
"در متن ارائه‌شده اشاره‌ای به این موضوع نشده است."

Contract Clauses:
{combined_context}

User Question:
{query}

Direct Answer in Persian:"""

    try:
        with pipeline_logger.track_latency("llm_generation", query=query):
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,  # صفر برای جلوگیری از پاسخ‌های متناقض و خلاقیت اضافی
                        "num_predict": 150,
                        "num_ctx": 1024
                    }
                },
                timeout=120
            )
            if response.status_code == 200:
                data = response.json()
                ans = data.get("response", "").strip()
                
                # اگر مدل پاسخ مثبت داد ولی جمله تناقض‌امیز را هم چسباند، بخش منفی را برش می‌زنیم
                if "در متن ارائه‌شده اشاره‌ای به این موضوع نشده است" in ans and len(ans) > 45:
                    ans = ans.replace("در متن ارائه‌شده اشاره‌ای به این موضوع نشده است.", "").strip()

                return ans if ans else "در متن ارائه‌شده اشاره‌ای به این موضوع نشده است."
            
            err_detail = response.text
            logger.error(f"Ollama HTTP {response.status_code}: {err_detail}")
            return f"⚠️ خطا در سرویس مدل زبانی (کد {response.status_code})"

    except Exception as e:
        logger.error(f"Error during LLM generation: {str(e)}")
        return f"⚠️ خطا در ارتباط با موتور Ollama: {str(e)}"