import json
import logging
import sys
import time
from contextlib import contextmanager

class JSONFormatter(logging.Formatter):
    """فرمت‌دهنده لاگ به صورت ساختاریافته JSON برای مانیتورینگ سازمانی"""
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        # اضافه کردن فیلدهای اضافه (مانند latency_ms یا query) در صورت وجود
        if hasattr(record, "extra_fields"):
            log_record.update(record.extra_fields)
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
    """کانتکست‌منیجر برای اندازه‌گیری زمان اجرای هر لایه از پایپ‌لاین"""
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
            extra["query"] = query
        if "error" in metrics:
            extra["error"] = metrics["error"]
            
        logger.info(f"Execution completed for stage: {stage_name}", extra={"extra_fields": extra})