import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # تنظیمات پایگاه داده برداری Qdrant
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "legal_contracts")

    # مدل‌های امبدینگ و بازچینی (Reranker)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    USE_E5_PREFIX: bool = os.getenv("USE_E5_PREFIX", "true").lower() in ("true", "1", "yes")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    # تنظیمات جداسازی Provider مدل زبانی
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").lower()
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.2")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # پارامترهای استنتاج
    LLM_NUM_CTX: int = int(os.getenv("LLM_NUM_CTX", 4096))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", 350))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.0))

    # تنظیمات امنیتی و محدودیت حجم فایل
    APP_ACCESS_PIN: str = os.getenv("APP_ACCESS_PIN", "").strip()
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", 15))


settings = Settings()