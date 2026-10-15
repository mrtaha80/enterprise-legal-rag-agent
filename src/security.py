import re
from typing import List, Dict, Any


class SecurityGuard:
    """
    لایه دفاعی برای مقابله با Prompt Injection، پاکسازی اطلاعات حساس (PII)
    و ایمن‌سازی لاگ‌های سیستمی (پشتیبانی از فراخوانی مستقیم کلاس و نمونه).
    """

    INJECTION_PATTERNS = [
        r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions",
        r"disregard\s+(?:all\s+)?(?:previous|prior)\s+rules",
        r"you\s+are\s+now\s+a",
        r"system\s+override",
        r"reveal\s+(?:the\s+)?system\s+prompt",
        r"developer\s+mode",
        r"دستورات\s+قبلی\s+را\s+نادیده",
        r"قوانین\s+قبلی\s+را\s+فراموش",
        r"تو\s+از\s+این\s+به\s+بعد",
        r"حالا\s+تو\s+نقش",
        r"پرامپت\s+سیستم\s+را\s+نشان\s+بده",
    ]

    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "phone": r"(?:\+98|0098|0)?9\d{9}\b",
        "national_id": r"\b\d{10}\b",
        "api_key": r"\b(?:sk-[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*)\b",
    }

    @classmethod
    def detect_prompt_injection(cls, text: str) -> bool:
        """بررسی وجود الگوهای مخرب پرامپت اینجکشن در متن (با نرمال‌سازی فاصله‌ها و نیم‌فاصله)"""
        if not text:
            return False

        normalized = text.replace("\u200c", " ").replace("ي", "ی").replace("ك", "ک").lower()
        normalized = re.sub(r"\s+", " ", normalized)

        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                return True
        return False

    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """ماسک کردن اطلاعات شخصی (PII) و کلیدهای حساس در متن و لاگ‌ها"""
        if not text:
            return ""

        sanitized = str(text)
        sanitized = re.sub(cls.PII_PATTERNS["email"], "[REDACTED_EMAIL]", sanitized)
        sanitized = re.sub(cls.PII_PATTERNS["phone"], "[REDACTED_PHONE]", sanitized)
        sanitized = re.sub(cls.PII_PATTERNS["national_id"], "[REDACTED_NATIONAL_ID]", sanitized)
        sanitized = re.sub(cls.PII_PATTERNS["api_key"], "[REDACTED_SECRET]", sanitized)
        return sanitized

    @classmethod
    def secure_log_payload(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """پاکسازی دیکشنری extra_fields در لاگ‌ها جهت جلوگیری از نشت اطلاعات حساس"""
        if not isinstance(data, dict):
            return {}

        secured = dict(data)
        sensitive_keys = ("query", "text", "content", "message", "error", "response", "prompt")

        for key, val in secured.items():
            if key in sensitive_keys and val is not None:
                secured[key] = cls.sanitize_text(str(val))
            elif isinstance(val, dict):
                secured[key] = cls.secure_log_payload(val)

        return secured

    @classmethod
    def sanitize_http_error(cls, raw_body: str, max_len: int = 200) -> str:
        """برش (Truncate) و ماسک‌سازی بدنه خطای HTTP جهت جلوگیری از ثبت اطلاعات حساس در لاگ"""
        if not raw_body:
            return "empty_error_body"
        clean = cls.sanitize_text(str(raw_body)).replace("\n", " ").strip()
        if len(clean) > max_len:
            return clean[:max_len] + "...[TRUNCATED]"
        return clean

    @classmethod
    def filter_and_sanitize_chunks(cls, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """حذف چانک‌های آلوده به پرامپت اینجکشن (Zero-Tolerance) و پاکسازی PII از چانک‌های سالم"""
        safe_chunks = []
        for chunk in chunks:
            content = chunk.get("content") or chunk.get("text", "")
            if cls.detect_prompt_injection(content):
                continue

            updated_chunk = dict(chunk)
            updated_chunk["content"] = cls.sanitize_text(content)
            safe_chunks.append(updated_chunk)

        return safe_chunks


security_guard = SecurityGuard()