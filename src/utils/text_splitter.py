"""基于段落的流式文本分块模块。

不依赖正则或章节标题匹配，而是按段落（空行分隔的自然段落）分组，
以固定段落数为一个逻辑块，相邻块之间保留重叠段落，保证上下文连续。
文件读取采用逐行流式方式，不一次性加载全文到内存。
"""

from typing import Generator, List, Optional

from loguru import logger

DEFAULT_ENCODINGS = ["utf-8", "gbk", "latin-1"]


def iter_paragraphs(
    file_path: str,
    encodings: Optional[List[str]] = None,
) -> Generator[str, None, None]:
    """逐行读取文件，按空行分割段落，yield 每个段落。

    连续的非空行属于同一段落，遇到空行时段落结束。
    文件以流式方式读取，不一次性加载到内存。
    """
    if encodings is None:
        encodings = DEFAULT_ENCODINGS

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                buffer: List[str] = []
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        buffer.append(stripped)
                    else:
                        if buffer:
                            yield "\n".join(buffer)
                            buffer = []
                if buffer:
                    yield "\n".join(buffer)
            return
        except UnicodeDecodeError:
            continue

    logger.error(f"无法解码文件: {file_path}")


def chunk_by_paragraphs(
    file_path: str,
    paragraphs_per_chunk: int = 30,
    overlap: int = 5,
    max_chunk_chars: int = 20000,
    encodings: Optional[List[str]] = None,
) -> List[str]:
    """基于段落数量自适应分块，相邻块之间保留重叠段落。

    Args:
        file_path: 文件路径
        paragraphs_per_chunk: 每块包含的最大段落数
        overlap: 相邻块之间重叠的段落数
        max_chunk_chars: 每块最大字符数（安全阀，防止超长段落撑爆）
        encodings: 尝试的编码列表

    Returns:
        分块后的文本列表
    """
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_chars = 0
    total_paragraphs = 0

    for para in iter_paragraphs(file_path, encodings):
        total_paragraphs += 1
        current_chunk.append(para)
        current_chars += len(para) + 2

        if len(current_chunk) >= paragraphs_per_chunk or current_chars >= max_chunk_chars:
            chunks.append("\n\n".join(current_chunk))
            if 0 < overlap < len(current_chunk):
                current_chunk = current_chunk[-overlap:]
                current_chars = sum(len(p) + 2 for p in current_chunk)
            else:
                current_chunk = []
                current_chars = 0

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    logger.info(
        f"段落分块: {total_paragraphs} 段落 → {len(chunks)} 块 "
        f"(每块 ≤{paragraphs_per_chunk} 段, 重叠 {overlap} 段, "
        f"上限 {max_chunk_chars} 字符)"
    )
    return chunks


def chunk_text_by_paragraphs(
    text: str,
    paragraphs_per_chunk: int = 30,
    overlap: int = 5,
    max_chunk_chars: int = 20000,
) -> List[str]:
    """对已加载的文本字符串进行段落分块。

    用于非文件来源的文本（如 CLI 直接传入的上下文）。
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_chars = 0

    for para in paragraphs:
        current_chunk.append(para)
        current_chars += len(para) + 2

        if len(current_chunk) >= paragraphs_per_chunk or current_chars >= max_chunk_chars:
            chunks.append("\n\n".join(current_chunk))
            if 0 < overlap < len(current_chunk):
                current_chunk = current_chunk[-overlap:]
                current_chars = sum(len(p) + 2 for p in current_chunk)
            else:
                current_chunk = []
                current_chars = 0

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    logger.info(
        f"文本段落分块: {len(paragraphs)} 段落 → {len(chunks)} 块 "
        f"(每块 ≤{paragraphs_per_chunk} 段, 重叠 {overlap} 段)"
    )
    return chunks
