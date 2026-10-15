import re
from typing import Dict, Any

class SecurityGuard:
    """ماژول امنیت، ماسک‌سازی اطلاعات هویتی و مالی (PII) و دفاع در برابر Prompt Injection"""
    
    # الگوهای رایج حملات Prompt Injection داخل اسناد یا ورودی‌ها
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|all)\s+instructions",
        r"forget\s+(previous|all)\s+instructions",
        r"system\s+override",
        r"you\s+are\s+now\s+(free|unrestricted)",
        r"act\s+as\s+an?\s+unrestricted",
        r"پیرو\s+دستورات\s+قبلی\s+را\s+فراموش\s+کن",
        r"دستورات\s+قبلی\s+را\s+نادیده\s+بگیر",
        r"تو\s+دیگر\s+محدودیتی\s+نداری"
    ]

    @classmethod
    def detect_prompt_injection(cls, text: str) -> bool:
        if not text:
            return False
        clean_text = text.lower()
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, clean_text, re.IGNORECASE):
                return True
        return False

    @staticmethod
    def sanitize_text(text: str) -> str:
        if not text:
            return ""
        
        # ماسک کردن کدهای ملی یا شماره‌های ۱۰ رقمی مشابه
        sanitized = re.sub(r'\b\d{10}\b', '[REDACTED_ID]', text)
        
        # ماسک کردن شماره تلفن‌های ایرانی
        sanitized = re.sub(r'09\d{9}', '[REDACTED_PHONE]', sanitized)
        
        # ماسک کردن ایمیل‌ها
        sanitized = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[REDACTED_EMAIL]', sanitized)
        
        return sanitized

    @classmethod
    def secure_log_payload(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """پاکسازی داده‌های حساس پیش از ثبت در لاگ سیستم"""
        secured = data.copy()
        if "query" in secured and secured["query"]:
            secured["query"] = cls.sanitize_text(secured["query"])
        if "text" in secured and secured["text"]:
            secured["text"] = cls.sanitize_text(secured["text"])
        return secured