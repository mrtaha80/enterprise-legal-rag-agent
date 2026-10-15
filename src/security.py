import re
from typing import Dict, Any

class SecurityGuard:
    """ماژول امنیت، ماسک‌سازی اطلاعات هویتی و مالی (PII) و پاکسازی لاگ‌ها"""
    
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