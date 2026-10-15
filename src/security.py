import re
from typing import List, Dict, Any
from src.logger import logger

class SecurityGuard:
    """لایه دفاعی پیشرفته برای مقابله با Prompt Injection و پاکسازی اطلاعات حساس (PII)"""
    
    def __init__(self):
        # الگوهای جامع‌تر و ساختاری برای شناسایی تلاش‌های پرامپت اینجکشن (انگلیسی و فارسی)
        self.injection_patterns = [
            r"ignore previous instructions",
            r"ignore all prior instructions",
            r"disregard previous rules",
            r"you are now a",
            r"system override",
            r"reveal system prompt",
            r"دستورات قبلی را نادیده",
            r"فراموش کن",
            r"تو از این به بعد",
            r"حالا تو نقش",
            r"پرامپت سیستم را نشان بده"
        ]
        
        # الگوهای ماسک‌سازی PII (کد ملی، شماره تماس، ایمیل)
        self.pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"(\+98|0)?9\d{9}",
            "national_id": r"\b\d{10}\b"
        }

    def detect_prompt_injection(self, text: str) -> bool:
        """بررسی وجود الگوهای مخرب پرامپت اینجکشن در متن"""
        if not text:
            return False
        
        text_lower = text.lower()
        for pattern in self.injection_patterns:
            if re.search(pattern, text_lower):
                logger.warning(f"Security Alert: Malicious prompt injection pattern detected: '{pattern}'")
                return True
        return False

    def sanitize_text(self, text: str) -> str:
        """ماسک کردن اطلاعات شخصی (PII) در متن"""
        if not text:
            return text
            
        sanitized = text
        for pii_type, pattern in self.pii_patterns.items():
            if pii_type == "email":
                sanitized = re.sub(pattern, "[REDACTED_EMAIL]", sanitized)
            elif pii_type == "phone":
                sanitized = re.sub(pattern, "[REDACTED_PHONE]", sanitized)
            elif pii_type == "national_id":
                sanitized = re.sub(pattern, "[REDACTED_NATIONAL_ID]", sanitized)
                
        return sanitized

    def filter_and_sanitize_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """فیلتر کردن چانک‌های آلوده به پرامپت اینجکشن و پاکسازی PII از چانک‌های امن"""
        safe_chunks = []
        dropped_count = 0

        for chunk in chunks:
            content = chunk.get("content", "")
            
            # اگر پرامپت اینجکشن شناسایی شد، کل چانک حذف می‌شود (Zero-Tolerance Policy)
            if self.detect_prompt_injection(content):
                dropped_count += 1
                logger.error("Security Action: Dropped a malicious chunk due to prompt injection attempt.")
                continue
            
            # در غیر این صورت، متن پاکسازی (Sanitize) می‌شود
            chunk["content"] = self.sanitize_text(content)
            safe_chunks.append(chunk)

        if dropped_count > 0:
            logger.warning(f"Security Summary: Dropped {dropped_count} malicious chunks during ingestion.")

        return safe_chunks

security_guard = SecurityGuard()