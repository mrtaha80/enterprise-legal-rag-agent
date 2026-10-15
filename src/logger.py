import json
import logging
import sys
import time
from contextlib import contextmanager
from src.security import SecurityGuard


class JSONFormatter(logging.Formatter):
    """فرمت‌دهنده لاگ ساختاریافته JSON همراه با ماسک‌سازی امنیتی خودکار (PII Masking)"""

    def format(self, record: logging.LogRecord) -> str:
        clean_message = SecurityGuard.sanitize_text(record.getMessage())

        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.module,
            "message": clean_message,
        }

        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            safe_extra = SecurityGuard.secure_log_payload(record.extra_fields)
            log_record.update(safe_extra)

        return json.dumps(log_record, ensure_ascii=False)


def setup_logger(name: str = "legal_rag") -> logging.Logger:
    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(logging.INFO)

    if not logger_instance.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger_instance.addHandler(handler)
    # فعال بودن propagate برای اینکه فریم‌ورک تست (pytest caplog) بتواند لاگ‌ها را دریافت کند
    logger_instance.propagate = True
    return logger_instance


logger = setup_logger()


@contextmanager
def trace_latency(stage_name: str, query: str = None):
    """کانتکست‌منیجر امن برای اندازه‌گیری زمان اجرا و ثبت لاگ بدون نشت داده"""
    start_time = time.perf_counter()
    metrics = {"status": "success"}
    try:
        yield metrics
    except Exception as exc:
        metrics["status"] = "failed"
        metrics["error"] = SecurityGuard.sanitize_http_error(str(exc))
        raise
    finally:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        extra = {
            "stage": stage_name,
            "latency_ms": elapsed_ms,
            "status": metrics["status"],
        }
        if query:
            extra["query"] = SecurityGuard.sanitize_text(query)
        if "error" in metrics:
            extra["error"] = metrics["error"]

        logger.info(f"Execution completed for stage: {stage_name}", extra={"extra_fields": extra})


class PipelineLoggerWrapper:
    @staticmethod
    def track_latency(stage: str, query: str = None):
        return trace_latency(stage, query)

    @staticmethod
    def info(*args, **kwargs):
        logger.info(*args, **kwargs)

    @staticmethod
    def warning(*args, **kwargs):
        logger.warning(*args, **kwargs)

    @staticmethod
    def error(*args, **kwargs):
        logger.error(*args, **kwargs)


pipeline_logger = PipelineLoggerWrapper()