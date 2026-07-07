from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
import tiktoken


class ContextCompressor:
    """上下文压缩器 - 支持多种策略"""

    def __init__(self, llm: ChatOpenAI, max_tokens: int = 8000, chunk_size: int = 8000, max_workers: int = 5, max_retries: int = 3):
        self.llm = llm
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.encoder = tiktoken.encoding_for_model("gpt-4")

    def should_compress(self, text: str) -> bool:
        token_count = len(self.encoder.encode(text))
        return token_count > self.max_tokens * 0.7

    def _invoke_with_retry(self, prompt: str, fallback: str = "") -> str:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                response = self.llm.invoke(prompt)
                return response.content
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                retryable = any(kw in err_str for kw in ["429", "503", "timeout", "connection", "overloaded"])
                if not retryable or attempt == self.max_retries - 1:
                    logger.error(f"LLM 调用失败: {e}")
                    break
                delay = 2 ** attempt
                logger.warning(f"LLM 调用失败，{delay}s 后第 {attempt + 1}/{self.max_retries} 次重试: {e}")
                time.sleep(delay)
        return fallback

    def compress(self, text: str, strategy: str = "map_reduce") -> str:
        if not self.should_compress(text):
            return text

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=int(self.chunk_size * 0.1),
            length_function=len,
        )
        docs = text_splitter.create_documents([text])
        logger.info(f"文本分割为 {len(docs)} 个 chunks (每个约 {self.chunk_size} 字符)")

        if strategy == "map_reduce":
            return self._map_reduce_summary(docs)
        elif strategy == "stuff":
            return self._stuff_summary(docs)
        elif strategy == "extractive":
            return self._extractive_summary(docs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _map_reduce_summary(self, docs: List[Document]) -> str:
        map_prompt = PromptTemplate.from_template("""
请用中文总结以下文本的核心内容，提取关键事实、数字和结论。
要求：保留所有关键实体（人名、地名、机构名）和重要数据。

文本：
{text}

总结：
""")

        def _summarize_chunk(idx: int, text: str) -> tuple:
            prompt = map_prompt.format(text=text)
            result = self._invoke_with_retry(prompt, fallback="（总结失败）")
            return idx, result

        map_results = [None] * len(docs)
        completed = 0
        total = len(docs)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {
                executor.submit(_summarize_chunk, i, doc.page_content): i
                for i, doc in enumerate(docs)
            }
            for future in as_completed(future_map):
                completed += 1
                try:
                    idx, result = future.result()
                    map_results[idx] = result
                    status = "完成" if result != "（总结失败）" else "失败"
                    logger.info(f"压缩进度: [{completed}/{total}] chunk {idx} {status}")
                except Exception as e:
                    logger.error(f"压缩进度: [{completed}/{total}] 失败: {e}")

        combined_summary = "\n\n".join([r for r in map_results if r])
        success_count = len([r for r in map_results if r and r != "（总结失败）"])
        logger.info(f"Map 阶段完成: {success_count}/{total} 成功")

        reduce_prompt = PromptTemplate.from_template("""
以下是多个片段的总结，请将它们整合成一个连贯的、结构化的完整总结。
去除重复信息，按逻辑顺序重新组织。

片段总结：
{text}

最终完整总结：
""")
        final_result = self._invoke_with_retry(
            reduce_prompt.format(text=combined_summary),
            fallback=combined_summary,
        )
        if final_result != combined_summary:
            logger.info("Reduce 阶段完成")
        else:
            logger.warning("Reduce 阶段失败，返回 Map 阶段拼接结果")
        return final_result

    def _stuff_summary(self, docs: List[Document]) -> str:
        combined = "\n\n".join([d.page_content for d in docs])
        stuff_prompt = PromptTemplate.from_template("""
请用中文总结以下文本的核心内容，提取关键事实、数字和结论。

文本：
{text}

总结：
""")
        return self._invoke_with_retry(stuff_prompt.format(text=combined), fallback=combined)

    def _extractive_summary(self, docs: List[Document]) -> str:
        combined = "\n\n".join([d.page_content for d in docs])
        extract_prompt = PromptTemplate.from_template("""
从以下文本中提取最重要的5-10个关键句子（原句摘抄），这些句子必须包含：
1. 核心论点或结论
2. 关键数据或事实
3. 重要实体关系

文本：
{text}

关键句子（保持原样，不要改写）：
""")
        return self._invoke_with_retry(extract_prompt.format(text=combined), fallback=combined)