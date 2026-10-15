import json
import logging
import sys
import time
from contextlib import contextmanager
from src.security import SecurityGuard

class JSONFormatter(logging.Formatter):
    """فرمت‌دهنده لاگ ساختاریافته همراه با ماسک‌سازی امنیتی خودکار (PII Masking)"""
    def format(self, record: logging.LogRecord) -> str:
        # پاکسازی پیام اصلی لاگ از اطلاعات حساس
        clean_message = SecurityGuard.sanitize_text(record.getMessage())
        
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.module,
            "message": clean_message,
        }
        
        # پاکسازی فیلدهای اضافه (مثل پرسش‌ها یا متون حقوقی) پیش از ثبت
        if hasattr(record, "extra_fields"):
            safe_extra = SecurityGuard.secure_log_payload(record.extra_fields)
            log_record.update(safe_extra)
            
        return json.dumps(log_record, ensure_ascii=False)

def setup_logger(name: str = "legal_rag") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger

logger = setup_logger()

@contextmanager
def trace_latency(stage_name: str, query: str = None):
    """کانتکست‌منیجر امن برای اندازه‌گیری زمان و ثبت لاگ بدون نشت داده"""
    start_time = time.perf_counter()
    metrics = {"status": "success"}
    try:
        yield metrics
    except Exception as exc:
        metrics["status"] = "failed"
        metrics["error"] = str(exc)
        raise
    finally:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        extra = {
            "stage": stage_name,
            "latency_ms": elapsed_ms,
            "status": metrics["status"]
        }
        if query:
            # ماسک کردن پرسش کاربر در لاگ‌های سیستمی
            extra["query"] = SecurityGuard.sanitize_text(query)
        if "error" in metrics:
            extra["error"] = SecurityGuard.sanitize_text(metrics["error"])
            
        logger.info(f"Execution completed for stage: {stage_name}", extra={"extra_fields": extra})

class PipelineLoggerWrapper:
    @staticmethod
    def track_latency(stage: str, query: str = None):
        return trace_latency(stage, query)
    
    @staticmethod
    def info(*args, **kwargs):
        logger.info(*args, **kwargs)

    @staticmethod
    def error(*args, **kwargs):
        logger.error(*args, **kwargs)

pipeline_logger = PipelineLoggerWrapper()