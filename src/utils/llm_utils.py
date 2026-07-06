import time
from typing import Optional, Tuple

from langchain_openai import ChatOpenAI
from loguru import logger

from src.core.config import settings


def create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.api_model,
        temperature=settings.openai_temperature,
        base_url=settings.api_base_url or None,
        api_key=settings.api_key or None,
        request_timeout=60,
        max_retries=0,
    )


def invoke_with_retry(
    llm: ChatOpenAI,
    prompt: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    fallback: str = "",
) -> Tuple[str, int]:
    try:
        response = llm.invoke(prompt)
        content = response.content
        tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens = response.usage_metadata.get("total_tokens", 0)
        elif hasattr(response, "response_metadata") and response.response_metadata:
            token_usage = response.response_metadata.get("token_usage", {})
            tokens = token_usage.get("total_tokens", 0)
        return content, tokens
    except Exception as e:
        error_str = str(e).lower()
        retryable = any(kw in error_str for kw in ["429", "503", "timeout", "connection", "overloaded"])

        for attempt in range(1, max_retries + 1):
            if not retryable:
                break

            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(f"LLM 调用失败 (可重试), {delay:.1f}s 后第 {attempt}/{max_retries} 次重试: {e}")

            time.sleep(delay)
            try:
                response = llm.invoke(prompt)
                content = response.content
                tokens = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    tokens = response.usage_metadata.get("total_tokens", 0)
                elif hasattr(response, "response_metadata") and response.response_metadata:
                    token_usage = response.response_metadata.get("token_usage", {})
                    tokens = token_usage.get("total_tokens", 0)
                logger.info(f"第 {attempt} 次重试成功")
                return content, tokens
            except Exception as retry_err:
                retry_str = str(retry_err).lower()
                if not any(kw in retry_str for kw in ["429", "503", "timeout", "connection", "overloaded"]):
                    logger.error(f"重试遇到不可恢复错误: {retry_err}")
                    break
                if attempt == max_retries:
                    logger.error(f"LLM 调用失败，已达最大重试次数 {max_retries}: {retry_err}")
                e = retry_err

        logger.error(f"LLM 调用最终失败: {e}")
        return fallback, 0