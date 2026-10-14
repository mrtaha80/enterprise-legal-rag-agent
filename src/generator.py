from src.config import settings

def generate_answer(query: str, retrieved_contexts: list[dict]) -> str:
    if not retrieved_contexts:
        return "با توجه به اسناد بارگذاری‌شده، هیچ بخش مرتبطی برای این پرسش یافت نشد."

    # اگر کلید معتبر ست شده باشد، از API استفاده می‌کند
    if settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("your_api"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)
            
            context_str = "\n\n---\n\n".join([
                f"[سند: {c['source']} | صفحه: {c['page']}]\n{c['text']}"
                for c in retrieved_contexts
            ])
            system_prompt = (
                "تو یک تحلیل‌گر ارشد اسناد حقوقی و مالی هستی.\n"
                "تنها منبع معتبر تو متن‌های ارائه‌شده در بخش Context هستند.\n"
                "دستورالعمل‌ها:\n"
                "۱. فقط و فقط به اطلاعات موجود در متن ارائه‌شده استناد کن.\n"
                "۲. در پایان هر بند، منبع و شماره صفحه را ذکر کن."
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nپرسش:\n{query}"}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"خطا در اتصال به API: {e}"

    # در صورت نبود کلید، پاسخ مستند بر مبنای چانک‌های برتر بازچینی‌شده نمایش داده می‌شود
    top_chunk = retrieved_contexts[0]
    formatted_response = (
        f"**پاسخ مستند بر اساس مفاد قرارداد:**\n\n"
        f"{top_chunk['text']}\n\n"
        f"📌 **ارجاع و منبع:** سند `{top_chunk['source']}` | **صفحه {top_chunk['page']}**\n\n"
        f"*(توجه: این پاسخ مستقیماً از موتور بازیابی و Re-ranker استخراج شده است؛ با قرار دادن API Key معتبر، متن توسط LLM بازنویسی می‌شود.)*"
    )
    return formatted_response