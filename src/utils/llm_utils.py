import json
import re
import time
from typing import Optional, Tuple, Type, TypeVar

from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel

from src.core.config import settings

T = TypeVar("T", bound=BaseModel)

_token_counter = None


def _estimate_tokens(text: str) -> int:
    """粗估 token 数（中文字符≈1.5 token，英文≈0.25 token/字符）"""
    if not text:
        return 0
    cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - cn_chars
    return int(cn_chars * 1.5 + other_chars * 0.25)


def safe_json_loads(content: str):
    """健壮地解析 LLM 返回的 JSON。

    处理以下常见情况：
    - markdown 代码块围栏（```json ... ```）
    - 前后多余文本/空白
    - 直接是数组或对象
    解析失败返回 None。
    """
    if not content:
        return None
    text = content.strip()
    if not text:
        return None

    # 去掉 markdown 代码块围栏（兼容闭合与未闭合两种情况）
    fence = re.search(r"```", text)
    if fence:
        text = text[fence.end():]
        closing = text.rfind("```")
        if closing > 0:
            text = text[:closing]
        text = text.strip()
        # 去掉残留的语言标记（如 json 紧跟在 ``` 后）
        text = re.sub(r"^[A-Za-z0-9_+-]+\s*\n", "", text)

    # 尝试直接解析（含去掉残留反引号的情况）
    candidates = [text, text.rstrip("`").rstrip()] if text else []
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

    return None


def create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.api_model,
        temperature=settings.openai_temperature,
        base_url=settings.api_base_url or None,
        api_key=settings.api_key or None,
        request_timeout=settings.llm_request_timeout,
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


_structured_disabled_until: dict = {}  # method → 失效时间戳，限时缓存避免重复 400
_STRUCTURED_DISABLE_TTL = 300  # 失败后跳过 5 分钟，之后重新尝试（投机解码可能已关闭）


def structured_extract(
    llm: ChatOpenAI,
    prompt: str,
    output_model: Type[T],
    max_retries: int = 1,
    base_delay: float = 1.0,
    back: bool = False,
) -> Tuple[Optional[T], int]:
    """用 with_structured_output 约束 LLM 返回 Pydantic 模型。

    优先用 function calling 约束；若模型不支持则回退到 json_mode + 手动解析。
    不支持的 method 会被缓存，后续调用直接跳过，避免重复 400。
    Returns:
        (model_instance, tokens) — 失败返回 (None, 0)
    """
    for method in ("function_calling", "json_mode"):
        if method in _structured_disabled_until and time.time() < _structured_disabled_until[method]:
            continue
        try:
            structured_llm = llm.with_structured_output(output_model, method=method)
        except Exception:
            _structured_disabled_until[method] = time.time() + _STRUCTURED_DISABLE_TTL
            continue

        for attempt in range(max_retries + 1):
            try:
                result = structured_llm.invoke(prompt)
                if isinstance(result, output_model):
                    return result, _estimate_tokens(prompt) + _estimate_tokens(result.model_dump_json())
                if isinstance(result, dict):
                    return output_model.model_validate(result), _estimate_tokens(prompt)
                logger.warning(f"结构化输出类型不符({method}): {type(result).__name__}, 重试 {attempt}/{max_retries}")
            except Exception as e:
                logger.error(e)
                err_str = str(e).lower()
                if "grammar" in err_str or "not support" in err_str or "400" in err_str:
                    logger.warning(f"模型暂时不支持 {method} 结构化输出，{_STRUCTURED_DISABLE_TTL}s 后重试")
                    _structured_disabled_until[method] = time.time() + _STRUCTURED_DISABLE_TTL
                    break
                retryable = any(kw in err_str for kw in ["429", "503", "timeout", "connection", "overloaded"])
                if not retryable or attempt == max_retries:
                    logger.warning(f"{method} 结构化输出失败: {e}")
                    break
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
    if back:
        # 最终回退：普通调用 + safe_json_loads
        logger.info("结构化输出不可用，回退到普通 JSON 解析")
        content, tokens = invoke_with_retry(
            llm, prompt,
            max_retries=max_retries,
            base_delay=base_delay,
            fallback="",
        )
        parsed = safe_json_loads(content)
        if parsed is not None:
            try:
                if isinstance(parsed, list):
                    if "items" in getattr(output_model, "model_fields", {}):
                        return output_model.model_validate({"items": parsed}), tokens
                    return None, tokens
                if isinstance(parsed, dict):
                    return output_model.model_validate(parsed), tokens
            except Exception as e:
                logger.warning(f"回退解析 model_validate 失败: {e}")
        return None, tokens
    return None, 0

