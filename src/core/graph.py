import json
import os
import time
import threading
from datetime import datetime
from typing import Literal, Tuple

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.types import Send

from loguru import logger

from src.agents.prompts import (
    INTENT_RECOGNITION_PROMPT,
    DIRECT_ANSWER_PROMPT,
    DEEP_ANALYSIS_SUPERVISOR_PROMPT,
    DEEP_ANALYSIS_WORKER_PROMPT,
    DEEP_ANALYSIS_QUALITY_PROMPT,
    DEEP_ANALYSIS_REDUCE_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_MERGE_PROMPT,
    TOOL_CALL_QUALITY_PROMPT,
    FILE_PATH_EXTRACTION_PROMPT,
)
from src.context.compressor import ContextCompressor
from src.core.config import settings
from src.core.state import AgentState, IntentType, WorkerState, EntityExtractionState, DeepAnalysisState
from src.utils.llm_utils import create_llm, invoke_with_retry
from src.utils.doc_parser import parse_file, parse_file_chunked
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===== 惰性初始化 =====
llm: ChatOpenAI = None
compressor: ContextCompressor = None
_worker_semaphore: threading.Semaphore = None


def _ensure_llm():
    global llm, compressor, _worker_semaphore
    if llm is None:
        llm = create_llm()
        compressor = ContextCompressor(
            llm,
            max_tokens=settings.max_context_tokens,
            chunk_size=settings.compression_chunk_size * 4,
            max_workers=settings.max_concurrent_workers,
            max_retries=settings.llm_max_retries,
        )
        _worker_semaphore = threading.Semaphore(settings.max_concurrent_workers)


def _invoke(prompt: str, fallback: str = "", node_name: str = "") -> Tuple[str, int]:
    _ensure_llm()
    content, tokens = invoke_with_retry(
        llm,
        prompt,
        max_retries=settings.llm_max_retries,
        base_delay=settings.llm_retry_base_delay,
        fallback=fallback,
    )
    if tokens > 0:
        logger.info(f"[{node_name}] tokens={tokens}")
    return content, tokens


def _record_step(state: AgentState, node_name: str, tokens: int = 0):
    step = f"{datetime.now().strftime('%H:%M:%S')} | {node_name} | tokens={tokens}"
    state.execution_steps.append(step)
    state.total_tokens_used += tokens


# ===== 意图识别 =====

import glob


def intent_recognition_node(state: AgentState) -> AgentState:
    if not state.original_input.strip():
        state.intent = IntentType.DIRECT_ANSWER
        logger.warning("用户输入为空，默认 direct_answer")
        _record_step(state, "intent_recognition")
        return state

    prompt = INTENT_RECOGNITION_PROMPT.format(question=state.original_input)
    raw, tokens = _invoke(prompt, fallback="direct_answer", node_name="intent_recognition")
    raw = raw.strip().lower()

    if "file_analysis" in raw:
        state.intent = IntentType.FILE_ANALYSIS
    elif "deep_analysis" in raw:
        state.intent = IntentType.DEEP_ANALYSIS
    elif "tool_call" in raw:
        state.intent = IntentType.TOOL_CALL
    else:
        state.intent = IntentType.DIRECT_ANSWER

    _record_step(state, "intent_recognition", tokens)
    logger.info(f"意图识别结果: {state.intent.value}")
    return state


def route_by_intent(state: AgentState) -> Literal["direct_answer_node", "supervisor", "compress_context", "file_reader"]:
    if state.intent == IntentType.DIRECT_ANSWER:
        return "direct_answer_node"
    elif state.intent == IntentType.TOOL_CALL:
        return "supervisor"
    elif state.intent == IntentType.FILE_ANALYSIS:
        return "file_reader"
    else:
        return "compress_context"


# ===== 文件查找与读取 =====

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".doc", ".html", ".htm"}


def _fuzzy_match_file(path: str) -> str:
    """模糊匹配文件：提取目录和文件名，在目录中找最接近的文件"""
    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    name_without_ext = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1].lower()

    if not directory:
        directory = "."
    if not os.path.isdir(directory):
        return None

    candidates = []
    for f in os.listdir(directory):
        f_path = os.path.join(directory, f)
        if not os.path.isfile(f_path):
            continue
        f_lower = f.lower()
        if name_without_ext.lower() in f_lower:
            candidates.append(f_path)
        elif f_lower in name_without_ext.lower():
            candidates.append(f_path)

    if not candidates:
        return None

    best = min(candidates, key=lambda c: abs(len(os.path.basename(c)) - len(filename)))
    return best


def file_reader_node(state: AgentState) -> AgentState:
    prompt = FILE_PATH_EXTRACTION_PROMPT.format(question=state.original_input)
    response, tokens = _invoke(prompt, node_name="file_path_extraction")

    paths = []
    try:
        parsed = json.loads(response)
        paths = parsed.get("paths", [])
    except (json.JSONDecodeError, TypeError):
        logger.warning("文件路径提取 JSON 解析失败")

    if not paths:
        logger.warning("未从用户输入中提取到文件路径，转为 deep_analysis")
        state.intent = IntentType.DEEP_ANALYSIS
        _record_step(state, "file_reader", tokens)
        return state

    logger.info(f"提取到文件路径: {paths}")

    file_paths = []
    for p in paths:
        p = p.strip().strip("'\"")
        p = os.path.expanduser(p)

        if os.path.isdir(p):
            for ext in SUPPORTED_EXTENSIONS:
                file_paths.extend(glob.glob(os.path.join(p, f"**/*{ext}"), recursive=True))
        elif os.path.isfile(p):
            file_paths.append(p)
        elif glob.glob(p, recursive=True):
            file_paths.extend([f for f in glob.glob(p, recursive=True) if os.path.isfile(f)])
        else:
            logger.warning(f"路径不存在，尝试模糊匹配: {p}")
            matched = _fuzzy_match_file(p)
            if matched:
                logger.info(f"  模糊匹配成功: {p} → {matched}")
                file_paths.append(matched)
            else:
                parent = os.path.dirname(p) or "."
                if os.path.isdir(parent):
                    similar = [f for f in os.listdir(parent) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]
                    logger.warning(f"  模糊匹配失败。目录 {parent} 下可用文件: {similar}")
                else:
                    logger.warning(f"  目录不存在: {parent}")

    if not file_paths:
        logger.warning("未找到有效文件，直接告知用户")
        state.direct_answer = f"未找到您指定的文件。请检查文件路径是否正确。\n\n您输入的路径: {paths}\n\n建议：\n1. 确认文件名拼写无误\n2. 使用绝对路径\n3. 检查文件是否存在"
        _record_step(state, "file_reader", tokens)
        return state

    logger.info(f"找到 {len(file_paths)} 个文件: {file_paths}")

    chunk_size = settings.compression_chunk_size * 10
    all_chunks = []
    for fp in file_paths:
        try:
            chunks = parse_file_chunked(fp, chunk_size=chunk_size)
            for chunk in chunks:
                all_chunks.append(f"=== {os.path.basename(fp)} ===\n{chunk}")
        except Exception as e:
            logger.error(f"解析文件失败 {fp}: {e}")

    if not all_chunks:
        logger.warning("所有文件解析结果为空，直接告知用户")
        state.direct_answer = f"已找到文件但解析结果为空。\n已找到文件: {file_paths}\n\n可能原因：\n1. 文件格式不支持\n2. 文件内容为空\n3. 文件加密或损坏"
        _record_step(state, "file_reader", tokens)
        return state

    state.raw_context = "\n\n".join(all_chunks)
    state.extracted_file_paths = file_paths
    logger.info(f"文件读取完成，共 {len(all_chunks)} 个文本块，总长度 {len(state.raw_context)} 字符")

    _record_step(state, "file_reader", tokens)
    return state


# ===== 直接回答 =====

def route_after_file_reader(state: AgentState) -> Literal["compress_context", "direct_answer_node"]:
    if state.direct_answer:
        return "direct_answer_node"
    return "compress_context"


def direct_answer_node(state: AgentState) -> AgentState:
    prompt = DIRECT_ANSWER_PROMPT.format(question=state.original_input)
    content, tokens = _invoke(prompt, fallback="抱歉，暂时无法回答。", node_name="direct_answer")
    state.direct_answer = content
    _record_step(state, "direct_answer", tokens)
    logger.info("直接回答完成")
    return state


# ===== 上下文压缩 =====

def compress_context_node(state: AgentState) -> AgentState:
    _ensure_llm()

    raw = state.raw_context

    if state.context_path and not raw:
        if not os.path.isfile(state.context_path):
            logger.error(f"文件不存在: {state.context_path}")
            state.compressed_context = ""
            _record_step(state, "compress_context")
            return state

        chunk_size = 20000
        try:
            chunks = parse_file_chunked(state.context_path, chunk_size=chunk_size)
        except Exception as e:
            logger.error(f"文件解析失败: {e}")
            state.compressed_context = ""
            _record_step(state, "compress_context")
            return state

        if not chunks:
            logger.error(f"文件内容为空: {state.context_path}")
            state.compressed_context = ""
            _record_step(state, "compress_context")
            return state
        elif len(chunks) == 1:
            raw = chunks[0]
        else:
            logger.info(f"文件较大，分 {len(chunks)} 块解析")
            raw = "\n\n".join(chunks)

    if not raw:
        raw = "\n".join([
            m.content for m in state.messages if isinstance(m, HumanMessage)
        ])

    if not raw or len(raw) < 100:
        state.compressed_context = raw
        _record_step(state, "compress_context")
        return state

    total_tokens = 0

    if compressor.should_compress(raw):
        logger.info(f"上下文过长，执行压缩... (原始长度: {len(raw)} 字符)")
        strategy = state.compression_strategy
        if strategy == "auto" or strategy == "AUTO":
            strategy = "map_reduce"
        compressed, tokens = _invoke_compressor(raw, strategy)
        total_tokens += tokens
        state.compressed_context = compressed
        state.compression_ratio = len(compressed) / len(raw)
        state.compression_strategy = "map_reduce"

        logger.info(f"压缩完成！压缩比: {state.compression_ratio:.2%}")
        logger.info(f"原始: {len(raw)} 字符 → 压缩后: {len(compressed)} 字符")

        state.messages.append(
            AIMessage(content=f"[系统] 已将长上下文压缩为 {len(compressed)} 字符的摘要")
        )
    else:
        state.compressed_context = raw
        logger.info("上下文长度适中，无需压缩")

    _record_step(state, "compress_context", total_tokens)
    return state


def _invoke_compressor(text: str, strategy: str) -> Tuple[str, int]:
    _ensure_llm()
    try:
        compressed = compressor.compress(text, strategy=strategy)
        return compressed, 0
    except Exception as e:
        logger.error(f"压缩失败: {e}")
        return text, 0


# ===== 任务拆解 =====

def supervisor_node(state: AgentState) -> AgentState:
    context_to_use = state.compressed_context or state.raw_context

    if state.intent in (IntentType.DEEP_ANALYSIS, IntentType.FILE_ANALYSIS) and context_to_use:
        prompt = DEEP_ANALYSIS_SUPERVISOR_PROMPT.format(
            context=context_to_use[:8000],
            question=state.original_input,
        )
    else:
        prompt = f"请根据用户需求执行：{state.original_input}"

    response, tokens = _invoke(prompt, node_name="supervisor")

    try:
        parsed = json.loads(response)
        state.task_plan = parsed
        state.entity_schema = parsed.get("entity_schema")
    except (json.JSONDecodeError, TypeError):
        logger.warning("任务拆解 JSON 解析失败，使用兜底计划")
        state.task_plan = {"tasks": [{"id": 1, "description": "全面分析", "assigned_to": "analyst_1", "role": "你是一个领域分析专家，擅长从给定角度进行深入分析。"}]}
        state.entity_schema = None

    tasks = state.task_plan.get("tasks", []) if state.task_plan else []
    schema_info = ""
    if state.entity_schema:
        schema_info = f" | 实体类型: {state.entity_schema.get('entity_type', '?')}, 属性: {state.entity_schema.get('attributes', [])}"
    _record_step(state, "supervisor", tokens)
    logger.info(f"任务拆解完成，共 {len(tasks)} 个子任务{schema_info}")
    for t in tasks:
        logger.info(f"  📋 {t.get('assigned_to', '?')}: {t.get('description', '?')}")

    return state


# ===== 路由：有 entity_schema 走两阶段，否则走旧流程 =====

def route_after_supervisor(state: AgentState):
    """有 entity_schema 走两阶段实体抽取，否则走旧 worker 流程"""
    if state.entity_schema:
        logger.info(f"route_after_supervisor: 走两阶段流程 (entity_schema={state.entity_schema})")
        result = route_to_entity_extractors(state)
        logger.info(f"route_to_entity_extractors 返回: {type(result).__name__}, 数量: {len(result) if isinstance(result, list) else 'N/A'}")
        return result
    else:
        logger.info("route_after_supervisor: 走旧 worker 流程 (无 entity_schema)")
        result = route_to_workers(state)
        logger.info(f"route_to_workers 返回: {type(result).__name__}, 数量: {len(result) if isinstance(result, list) else 'N/A'}")
        return result


# ===== 阶段1：实体抽取（Map 并行） =====

def route_to_entity_extractors(state: AgentState):
    context_to_use = state.compressed_context or state.raw_context
    if not context_to_use or not state.entity_schema:
        return "entity_merge"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.compression_chunk_size * 4,
        chunk_overlap=int(settings.compression_chunk_size * 0.4),
        length_function=len,
    )
    chunks = splitter.split_text(context_to_use)
    logger.info(f"实体抽取：上下文分为 {len(chunks)} 个 chunk")

    sends = []
    for i, chunk in enumerate(chunks):
        extraction_state = EntityExtractionState(
            chunk_index=i,
            text=chunk,
            entity_type=state.entity_schema.get("entity_type", "实体"),
            attributes=", ".join(state.entity_schema.get("attributes", [])),
        )
        sends.append(Send("entity_extraction", extraction_state))
    logger.info(f"分发 {len(sends)} 个并行实体抽取 Worker")
    return sends


def entity_extraction_node(state: EntityExtractionState) -> dict:
    _ensure_llm()
    _worker_semaphore.acquire()
    try:
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            entity_type=state.entity_type,
            attributes=state.attributes,
            text=state.text,
        )
        logger.info(f"  🔍 chunk {state.chunk_index} 正在抽取实体...")
        start = time.time()

        content, tokens = invoke_with_retry(
            llm,
            prompt,
            max_retries=settings.llm_max_retries,
            base_delay=settings.llm_retry_base_delay,
            fallback="[]",
        )

        entities = []
        try:
            entities = json.loads(content)
            if not isinstance(entities, list):
                entities = []
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"  chunk {state.chunk_index} 实体 JSON 解析失败")
            entities = []

        elapsed = time.time() - start
        logger.info(f"  ✅ chunk {state.chunk_index} 抽取完成 ({elapsed:.1f}s, {len(entities)} 个实体)")
        return {"extracted_entities": entities}
    finally:
        _worker_semaphore.release()


# ===== 阶段1.5：实体合并 =====

def entity_merge_node(state: AgentState) -> AgentState:
    if not state.extracted_entities:
        logger.warning("无实体抽取结果，跳过合并")
        state.merged_entities = []
        _record_step(state, "entity_merge")
        return state

    all_entities = []
    for ent_list in state.extracted_entities:
        if isinstance(ent_list, list):
            all_entities.extend(ent_list)
        elif isinstance(ent_list, dict):
            all_entities.append(ent_list)

    logger.info(f"实体合并：共 {len(all_entities)} 个原始实体")

    if len(all_entities) > 200:
        batch_size = 200
        merged_batches = []
        for i in range(0, len(all_entities), batch_size):
            batch = all_entities[i:i + batch_size]
            merged = _merge_entities_batch(batch, state.entity_schema)
            merged_batches.extend(merged)
        state.merged_entities = merged_batches
    else:
        state.merged_entities = _merge_entities_batch(all_entities, state.entity_schema)

    logger.info(f"实体合并完成：{len(state.merged_entities)} 个合并后实体")
    _record_step(state, "entity_merge")
    return state


def _merge_entities_batch(entities: list, schema: dict) -> list:
    entity_type = schema.get("entity_type", "实体") if schema else "实体"
    prompt = ENTITY_MERGE_PROMPT.format(
        entity_type=entity_type,
        entities=json.dumps(entities, ensure_ascii=False),
    )
    content, tokens = _invoke(prompt, fallback="[]", node_name="entity_merge")
    try:
        merged = json.loads(content)
        if isinstance(merged, list):
            return merged
    except (json.JSONDecodeError, TypeError):
        logger.warning("实体合并 JSON 解析失败，返回原始实体")
    return entities


# ===== 阶段2：深度分析（Reduce 并行） =====

def route_to_deep_analysts(state: AgentState):
    if not state.merged_entities:
        return "aggregate"

    context_to_use = state.compressed_context or state.raw_context
    tasks = state.task_plan.get("tasks", []) if state.task_plan else []

    sends = []
    for i, entity in enumerate(state.merged_entities):
        task = tasks[i % len(tasks)] if tasks else {}
        role = task.get("role", "你是一个领域分析专家，擅长深入分析。")

        entity_info = json.dumps(entity, ensure_ascii=False, indent=2)
        analysis_state = DeepAnalysisState(
            agent_id=f"analyst_{i + 1}",
            role=role,
            entity_info=entity_info,
            context=context_to_use[:8000],
            question=state.original_input,
        )
        sends.append(Send("deep_analysis", analysis_state))

    logger.info(f"分发 {len(sends)} 个并行深度分析 Worker")
    return sends


def deep_analysis_node(state: DeepAnalysisState) -> dict:
    _ensure_llm()
    _worker_semaphore.acquire()
    try:
        prompt = DEEP_ANALYSIS_REDUCE_PROMPT.format(
            role=state.role,
            entity_info=state.entity_info,
            context=state.context,
            question=state.question,
        )
        logger.info(f"  🤖 {state.agent_id} 正在深度分析...")
        start = time.time()

        content, tokens = invoke_with_retry(
            llm,
            prompt,
            max_retries=settings.llm_max_retries,
            base_delay=settings.llm_retry_base_delay,
            fallback="分析失败，无结果。",
        )

        elapsed = time.time() - start
        logger.info(f"  ✅ {state.agent_id} 深度分析完成 ({elapsed:.1f}s, tokens={tokens})")
        return {"worker_results": {state.agent_id: content}}
    finally:
        _worker_semaphore.release()


# ===== 旧流程：直接 worker 分发（无 entity_schema 时） =====

def route_to_workers(state: AgentState):
    context_to_use = state.compressed_context or state.raw_context
    tasks = state.task_plan.get("tasks", []) if state.task_plan else []

    if not tasks:
        return "aggregate"

    sends = []
    for t in tasks:
        worker_state = WorkerState(
            agent_id=t.get("assigned_to", "analyst_1"),
            role=t.get("role", "你是一个领域分析专家，擅长从给定角度进行深入分析。"),
            task_description=t.get("description", ""),
            context=context_to_use,
        )
        sends.append(Send("worker", worker_state))
    logger.info(f"分发 {len(sends)} 个并行 Worker (并发上限: {settings.max_concurrent_workers})")
    return sends


def worker_agent_node(state: WorkerState) -> dict:
    _ensure_llm()
    _worker_semaphore.acquire()
    try:
        prompt = DEEP_ANALYSIS_WORKER_PROMPT.format(
            role=state.role,
            context=state.context,
            task_description=state.task_description,
        )
        logger.info(f"  🤖 {state.agent_id} 正在分析: {state.task_description}")
        start = time.time()

        content, tokens = invoke_with_retry(
            llm,
            prompt,
            max_retries=settings.llm_max_retries,
            base_delay=settings.llm_retry_base_delay,
            fallback="分析失败，无结果。",
        )

        elapsed = time.time() - start
        logger.info(f"  ✅ {state.agent_id} 分析完成 ({elapsed:.1f}s, tokens={tokens})")
        return {"worker_results": {state.agent_id: content}}
    finally:
        _worker_semaphore.release()


# ===== 汇总节点 =====

def aggregate_node(state: AgentState) -> AgentState:
    if state.worker_results:
        logger.info(f"汇总 {len(state.worker_results)} 个 Worker 结果")
    else:
        logger.warning("无 Worker 结果汇总")

    _record_step(state, "aggregate")
    return state


# ===== 质量自检 =====

def quality_harness_node(state: AgentState) -> AgentState:
    if state.intent in (IntentType.DEEP_ANALYSIS, IntentType.FILE_ANALYSIS):
        prompt = DEEP_ANALYSIS_QUALITY_PROMPT.format(
            question=state.original_input,
            tasks=json.dumps(state.task_plan, ensure_ascii=False) if state.task_plan else "{}",
            results=json.dumps(state.worker_results, ensure_ascii=False),
        )
    elif state.intent == IntentType.TOOL_CALL:
        prompt = TOOL_CALL_QUALITY_PROMPT.format(
            question=state.original_input,
            results=json.dumps(state.worker_results, ensure_ascii=False),
        )
    else:
        logger.info("direct_answer 无需质量评估")
        state.quality_score = 1.0
        _record_step(state, "quality_harness")
        return state

    response, tokens = _invoke(prompt, node_name="quality_harness")
    if not response:
        logger.warning("质量评估 LLM 调用失败，跳过评分")
        state.quality_score = 0.5
        _record_step(state, "quality_harness")
        return state

    try:
        scores = json.loads(response)
        state.quality_score = scores.get("overall", 0.0)
        state.quality_details = scores
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"质量评估 JSON 解析失败: {response[:100]}")
        state.quality_score = 0.5

    _record_step(state, "quality_harness", tokens)
    logger.info(f"质量评分: {state.quality_score:.2f}")
    return state


def should_retry(state: AgentState) -> Literal["supervisor", "finalize"]:
    if state.quality_score is not None and state.quality_score < settings.quality_threshold:
        if state.retry_count < state.max_retries:
            state.retry_count += 1
            logger.warning(f"质量分 {state.quality_score:.2f} < {settings.quality_threshold}，第 {state.retry_count} 次重试...")
            return "supervisor"
        else:
            logger.warning(f"达到最大重试次数 {state.max_retries}，强制结束")
            return "finalize"
    return "finalize"


# ===== 构建 Graph =====

def build_agent_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("intent_recognition", intent_recognition_node)
    workflow.add_node("direct_answer_node", direct_answer_node)
    workflow.add_node("file_reader", file_reader_node)
    workflow.add_node("compress_context", compress_context_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("entity_extraction", entity_extraction_node)
    workflow.add_node("entity_merge", entity_merge_node)
    workflow.add_node("deep_analysis", deep_analysis_node)
    workflow.add_node("worker", worker_agent_node)
    workflow.add_node("aggregate", aggregate_node)
    workflow.add_node("quality_harness", quality_harness_node)

    workflow.set_entry_point("intent_recognition")

    workflow.add_conditional_edges(
        "intent_recognition",
        route_by_intent,
        {
            "direct_answer_node": "direct_answer_node",
            "supervisor": "supervisor",
            "compress_context": "compress_context",
            "file_reader": "file_reader",
        }
    )

    workflow.add_edge("direct_answer_node", END)
    workflow.add_conditional_edges(
        "file_reader",
        route_after_file_reader,
        {
            "compress_context": "compress_context",
            "direct_answer_node": "direct_answer_node",
        }
    )
    workflow.add_edge("compress_context", "supervisor")

    # supervisor → 根据是否有 entity_schema 分流
    # 有：并行 entity_extraction → entity_merge → 并行 deep_analysis → aggregate
    # 无：并行 worker → aggregate
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        [["entity_extraction", "worker"], "entity_merge", "aggregate"],
    )
    workflow.add_edge("entity_extraction", "entity_merge")
    workflow.add_conditional_edges(
        "entity_merge",
        route_to_deep_analysts,
        [["deep_analysis"], "aggregate"],
    )
    workflow.add_edge("deep_analysis", "aggregate")
    workflow.add_edge("worker", "aggregate")

    workflow.add_edge("aggregate", "quality_harness")

    workflow.add_conditional_edges(
        "quality_harness",
        should_retry,
        {
            "supervisor": "supervisor",
            "finalize": END,
        }
    )

    return workflow.compile()