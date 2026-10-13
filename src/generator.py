from openai import OpenAI
from src.config import settings

client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL
)

def generate_answer(query: str, retrieved_contexts: list[dict]) -> str:
    if not retrieved_contexts:
        return "با توجه به اسناد بارگذاری‌شده، هیچ بخش مرتبطی برای این پرسش یافت نشد."

    context_str = "\n\n---\n\n".join([
        f"[سند: {c['source']} | صفحه: {c['page']}]\n{c['text']}"
        for c in retrieved_contexts
    ])

    system_prompt = \"\"\"تو یک تحلیل‌گر ارشد اسناد حقوقی و مالی هستی.
تنها منبع معتبر تو متن‌های ارائه‌شده در بخش Context هستند.
دستورالعمل‌ها:
۱. فقط و فقط به اطلاعات موجود در متن ارائه‌شده استناد کن. حدس نزن.
۲. پاسخ را شفاف، ساختاریافته و با لحن رسمی بنویس.
۳. در پایان هر بند یا انتهای پاسخ، منبع و شماره صفحه را دقیقاً ذکر کن (مثلاً: سند X، صفحه Y).
۴. اگر پاسخ در متن نبود، بگو: «با توجه به اسناد بارگذاری‌شده، پاسخی برای این سوال یافت نشد.»\"\"\"

    user_message = f"Context:\n{context_str}\n\nپرسش:\n{query}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content
