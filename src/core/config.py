from typing import Optional
import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

def _pop_proxy() -> None:
    for _key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(_key, None)
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault('no_proxy', "*")

_pop_proxy()


class Settings(BaseSettings):
    """统一配置管理"""

    # LLM API
    api_base_url: str = ""
    api_key: str = ""
    api_model: str = "gpt-4-turbo"
    openai_temperature: float = 0.0
    llm_request_timeout: int = 60
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 1.0
    max_concurrent_workers: int = 5

    # LangSmith（可观测性）
    langsmith_api_key: Optional[str] = ""
    langsmith_project: str = "multi-agent-system"
    langsmith_tracing: bool = False

    # Context Compression
    max_context_tokens: int = 8000
    compression_chunk_size: int = 2000
    compression_chunk_overlap: int = 200
    compression_strategy: str = "map_reduce"  # map_reduce | stuff | extractive

    # Agent Settings
    max_retries: int = 3
    quality_threshold: float = 0.7

    # Memory
    vector_db_path: str = "./data/vector_db"
    embedding_model: str = "text-embedding-3-small"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


settings = Settings()

# LangSmith 环境变量注入
if settings.langsmith_tracing and settings.langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")