import glob
import json
import os
import time
import threading
from datetime import datetime
from typing import Tuple

from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.types import Send

from loguru import logger

from src.agents.prompts import (
    CHARACTER_EXTRACTION_PROMPT,
    CHARACTER_EVOLUTION_PROMPT,
    CHARACTER_CARD_PROMPT,
    CHARACTER_QUALITY_PROMPT,
    ALIAS_CLEANUP_PROMPT,
    FILE_PATH_EXTRACTION_PROMPT,
)
from src.core.config import settings
from src.core.state import AgentState, EntityExtractionState, CharacterCardState
from src.utils.llm_utils import create_llm, invoke_with_retry, safe_json_loads
from src.utils.doc_parser import parse_file_paragraph_chunked
from src.utils.voice_mapper import VOICE_MAPPING_RULES_TEXT

# ===== 惰性初始化 =====
llm = None
_worker_semaphore: threading.Semaphore = None


def _ensure_llm():
    global llm, _worker_semaphore
    if llm is None:
        llm = create_llm()
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


# ===== 文件读取 =====

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".doc", ".html", ".htm", ".epub"}


def _fuzzy_match_file(path: str) -> str:
    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    name_without_ext = os.path.splitext(filename)[0]

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
    """从用户输入中解析小说文件路径，按段落流式分块，保留章节顺序。"""
    prompt = FILE_PATH_EXTRACTION_PROMPT.format(question=state.original_input)
    response, tokens = _invoke(prompt, fallback=json.dumps({"paths": [state.context_path]}), node_name="file_path_extraction")

    paths = []
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            paths = parsed.get("paths", [])
        elif isinstance(parsed, list):
            paths = parsed
    except (json.JSONDecodeError, TypeError):
        logger.warning("文件路径提取 JSON 解析失败")
        if state.context_path:
            paths = [state.context_path]

    file_paths = []
    for p in paths:
        p = (p or "").strip().strip("'\"")
        if not p:
            continue
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            for ext in SUPPORTED_EXTENSIONS:
                file_paths.extend(glob.glob(os.path.join(p, f"**/*{ext}"), recursive=True))
        elif os.path.isfile(p):
            file_paths.append(p)
        elif glob.glob(p, recursive=True):
            file_paths.extend([f for f in glob.glob(p, recursive=True) if os.path.isfile(f)])
        else:
            matched = _fuzzy_match_file(p)
            if matched:
                logger.info(f"  模糊匹配成功: {p} → {matched}")
                file_paths.append(matched)
            else:
                logger.warning(f"路径不存在且无法匹配: {p}")

    if not file_paths:
        state.direct_answer = f"未找到小说文件。请提供正确的文件路径。您输入的路径: {paths}"
        _record_step(state, "file_reader", tokens)
        return state

    logger.info(f"找到 {len(file_paths)} 个小说文件: {file_paths}")

    all_chunks = []
    for fp in file_paths:
        try:
            chunks = parse_file_paragraph_chunked(
                fp,
                paragraphs_per_chunk=settings.paragraphs_per_chunk,
                overlap=settings.paragraph_overlap,
                max_chunk_chars=settings.max_chunk_chars,
            )
            for chunk in chunks:
                all_chunks.append(f"=== {os.path.basename(fp)} ===\n{chunk}")
        except Exception as e:
            logger.error(f"解析小说文件失败 {fp}: {e}")

    if not all_chunks:
        state.direct_answer = f"小说文件解析结果为空。\n已找到文件: {file_paths}"
        _record_step(state, "file_reader", tokens)
        return state

    # 可选：只处理前 N 个段落块（测试用/快速验证，默认处理全部）
    limit = settings.parse_only_chunks
    if limit > 0 and len(all_chunks) > limit:
        logger.info(f"PARSE_ONLY_CHUNKS 生效：仅处理前 {limit}/{len(all_chunks)} 个段落块")
        all_chunks = all_chunks[:limit]

    state.text_chunks = all_chunks
    state.extracted_file_paths = file_paths
    state.is_character_analysis = True
    total_chars = sum(len(c) for c in all_chunks)
    logger.info(f"小说读取完成，共 {len(all_chunks)} 个段落块，总长度 {total_chars} 字符（未拼接全文）")

    _record_step(state, "file_reader", tokens)
    return state


def route_after_file_reader(state: AgentState):
    if state.direct_answer:
        logger.warning("文件读取失败，直接返回提示")
        return "direct_answer_node"
    # 成功读取：按段落块并行分发角色抽取
    if not state.text_chunks:
        state.direct_answer = "小说内容为空，无法生成角色卡。"
        return "direct_answer_node"
    sends = []
    for i, chunk in enumerate(state.text_chunks):
        extraction_state = EntityExtractionState(chunk_index=i, text=chunk)
        sends.append(Send("character_extraction", extraction_state))
    logger.info(f"分发 {len(sends)} 个并行角色抽取 Worker")
    return sends


def direct_answer_node(state: AgentState) -> AgentState:
    state.direct_answer = state.direct_answer or "无法解析该小说。"
    _record_step(state, "direct_answer")
    logger.warning("文件读取失败，直接返回提示")
    return state


# ===== 阶段1：角色抽取（Map 并行） =====
# 直接按段落块并行抽取角色，跳过 supervisor（角色 schema 固定）+ 上下文压缩
# （抽取基于原文块而非压缩摘要，保证不丢失角色细节；source_chunks 保留时间线）

def character_extraction_node(state: EntityExtractionState) -> dict:
    _ensure_llm()
    _worker_semaphore.acquire()
    try:
        prompt = CHARACTER_EXTRACTION_PROMPT.format(
            chunk_index=state.chunk_index,
            text=state.text,
        )
        logger.info(f"  🔍 chunk {state.chunk_index} 正在抽取角色...")
        start = time.time()

        content, tokens = invoke_with_retry(
            llm,
            prompt,
            max_retries=settings.llm_max_retries,
            base_delay=settings.llm_retry_base_delay,
            fallback="[]",
        )

        entities = []
        parsed = safe_json_loads(content)
        if isinstance(parsed, list):
            entities = parsed
        else:
            logger.warning(f"  chunk {state.chunk_index} 角色 JSON 解析失败，跳过")
        entities = [e for e in entities if isinstance(e, dict)]

        # 归一化：把 personality / emotion 的 list 形式转成顿号分隔字符串
        for e in entities:
            for k in ("personality", "emotion", "age"):
                if isinstance(e.get(k), list):
                    joined = "、".join(str(x).strip() for x in e[k] if str(x).strip())
                    e[k] = joined or ""

        elapsed = time.time() - start
        logger.info(f"  ✅ chunk {state.chunk_index} 抽取完成 ({elapsed:.1f}s, {len(entities)} 个角色)")

        for ent in entities:
            if isinstance(ent, dict):
                ent["source_chunks"] = [state.chunk_index]

        return {"extracted_entities": entities}
    finally:
        _worker_semaphore.release()


# ===== 阶段2：角色合并 =====

def character_merge_node(state: AgentState) -> AgentState:
    if not state.extracted_entities:
        logger.warning("无角色抽取结果，跳过合并")
        state.merged_entities = []
        _record_step(state, "character_merge")
        return state

    all_entities = []
    for ent_list in state.extracted_entities:
        if isinstance(ent_list, list):
            all_entities.extend(ent_list)
        elif isinstance(ent_list, dict):
            all_entities.append(ent_list)

    logger.info(f"角色合并：共 {len(all_entities)} 个原始角色")

    state.merged_entities = _merge_character_entities(all_entities)
    logger.info(f"角色合并完成：共 {len(state.merged_entities)} 个合并后角色")
    _record_step(state, "character_merge")
    return state


def _clean_aliases_with_llm(char_map: dict) -> None:
    """用 LLM 清洗所有角色的别名，剔除通用称呼/他人名字/描述性词语。

    原地修改 char_map 中每个角色的 aliases。
    """
    char_list = [{"name": c["name"], "aliases": list(c["aliases"])} for c in char_map.values()]
    if not char_list:
        return

    prompt = ALIAS_CLEANUP_PROMPT.format(
        characters=json.dumps(char_list, ensure_ascii=False),
    )
    try:
        content, _ = _invoke(prompt, fallback=json.dumps(char_list, ensure_ascii=False), node_name="alias_cleanup")
        cleaned = safe_json_loads(content)
        if not isinstance(cleaned, list):
            return
        name_to_aliases = {}
        for item in cleaned:
            if isinstance(item, dict) and item.get("name"):
                name_to_aliases[item["name"]] = item.get("aliases", []) or []
        for c in char_map.values():
            if c["name"] in name_to_aliases:
                kept = [a for a in name_to_aliases[c["name"]] if isinstance(a, str) and a and a != c["name"]]
                c["aliases"] = list(dict.fromkeys(kept))
    except Exception as e:
        logger.warning(f"别名清洗失败，保留原始别名: {e}")


def _merge_character_entities(all_entities: list) -> list:
    """合并角色：按名称+别名交叉聚合，保留每块的属性快照（timeline）。"""
    if not all_entities:
        return []

    # ---- Pass 1: 按精确 name 聚合，同时收集 timeline ----
    char_map: dict = {}
    for ent in all_entities:
        if not isinstance(ent, dict):
            continue
        name = (ent.get("name") or "").strip()
        if not name:
            continue
        if name not in char_map:
            char_map[name] = {
                "name": name,
                "aliases": [],
                "gender": "unknown",
                "age": "",
                "personality": "",
                "alignment": "neutral",
                "appearance": "",
                "background": "",
                "emotion": "",
                "source_chunks": [],
                "timeline": [],
            }
        target = char_map[name]

        for key in ["aliases", "gender", "age", "personality", "alignment", "appearance", "background", "emotion"]:
            val = ent.get(key)
            if not val:
                continue
            if key == "aliases":
                if isinstance(val, list):
                    for a in val:
                        a = (a or "").strip()
                        if a and a not in target["aliases"] and a != name:
                            target["aliases"].append(a)
            elif not _is_val_set(target[key]):
                target[key] = val
            elif key in ("personality", "emotion") and isinstance(val, str):
                for part in val.split("、"):
                    part = part.strip()
                    if part and part not in target[key].split("、"):
                        target[key] = target[key] + "、" + part

        if isinstance(ent.get("source_chunks"), list):
            for c in ent["source_chunks"]:
                if c not in target["source_chunks"]:
                    target["source_chunks"].append(c)

        # timeline 快照：保留每块的属性值 + chunk_index
        chunk_idx = (ent.get("source_chunks") or [None])[0]
        snapshot = {"chunk_index": chunk_idx}
        for key in ("age", "personality", "alignment", "emotion"):
            v = ent.get(key)
            if v:
                snapshot[key] = v
        target["timeline"].append(snapshot)

    # ---- Pass 1.5: LLM 清洗别名（剔除通用称呼、他人名字、描述性词语）----
    _clean_aliases_with_llm(char_map)

    # ---- Pass 2: 别名交叉合并 ----
    # 过滤通用称呼，避免两个角色因共有"公子""先生"等被误连
    GENERIC_TITLES = {
        "先生", "小姐", "公子", "兄台", "前辈", "阁下", "那位", "此人",
        "大叔", "大娘", "姑娘", "少年", "少女", "大人", "殿下", "陛下",
        "师父", "师姐", "师兄", "师弟", "师妹", "道友", "掌柜", "掌门",
        "本才子", "本座", "在下", "鄙人", "区区", "小可", "老夫", "老朽",
        "小妞", "辣货", "漂亮小妞", "死人妖", "人妖公子", "西贝货",
    }

    def _is_specific_alias(a: str) -> bool:
        a = (a or "").strip()
        if not a or len(a) <= 1:
            return False
        if a in GENERIC_TITLES:
            return False
        return True

    parent = {name: name for name in char_map}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # 对每对角色，双向确认才合并：A.name ∈ B.aliases 且 B.name ∈ A.aliases
    names = list(char_map.keys())
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            aliases_a = set(char_map[name_a]["aliases"])
            aliases_b = set(char_map[name_b]["aliases"])
            if (name_a in aliases_b and name_b in aliases_a) or name_a == name_b:
                union(name_a, name_b)

    # 按 canonical 分组
    groups: dict = {}
    for name in names:
        root = find(name)
        groups.setdefault(root, []).append(name)

    # 合并同组角色
    merged = []
    for group_names in groups.values():
        primary = char_map[group_names[0]]
        for other_name in group_names[1:]:
            other = char_map[other_name]
            primary["aliases"].extend(a for a in other["aliases"] if a not in primary["aliases"] and a != primary["name"])
            for key in ["gender", "age", "personality", "alignment", "appearance", "background", "emotion"]:
                if not _is_val_set(primary[key]) and _is_val_set(other[key]):
                    primary[key] = other[key]
            for c in other["source_chunks"]:
                if c not in primary["source_chunks"]:
                    primary["source_chunks"].append(c)
            primary["timeline"].extend(other["timeline"])

        primary["source_chunks"].sort()
        primary["aliases"] = [a for a in dict.fromkeys(primary["aliases"]) if _is_specific_alias(a) and a != primary["name"]]
        primary["timeline"].sort(key=lambda t: (t.get("chunk_index") is None, t.get("chunk_index")))
        merged.append(primary)

    return merged


def _is_val_set(val) -> bool:
    if val is None:
        return False
    if isinstance(val, str):
        return bool(val.strip())
    return bool(val)


# ===== 阶段3：角色转变检测 =====

def character_evolution_node(state: AgentState) -> AgentState:
    """检测角色是否有显著转变，拆分为多个时期。"""
    if not state.merged_entities:
        logger.warning("无合并角色，跳过转变检测")
        _record_step(state, "character_evolution")
        return state

    evolution = {}
    total_chunks = max(len(state.text_chunks), 1)

    for ent in state.merged_entities:
        if not isinstance(ent, dict):
            continue
        name = ent.get("name") or "unknown"
        source_chunks = ent.get("source_chunks") or []
        if not isinstance(source_chunks, list):
            source_chunks = []

        # 规则过滤：戏份太少或无跨度 → 不检测转变
        if len(source_chunks) < 3:
            evolution[name] = [{"period_name": "全程", "source_chunks": source_chunks, **ent}]
            continue

        span = max(source_chunks) - min(source_chunks) + 1
        if span <= total_chunks * 0.2:
            evolution[name] = [{"period_name": "全程", "source_chunks": source_chunks, **ent}]
            continue

        entity_info = json.dumps({
            "name": ent.get("name"),
            "aliases": ent.get("aliases"),
            "gender": ent.get("gender"),
            "timeline": ent.get("timeline", []),
        }, ensure_ascii=False)
        prompt = CHARACTER_EVOLUTION_PROMPT.format(
            name=name,
            entity_info=entity_info,
            source_chunks=str(source_chunks),
        )
        content, _ = _invoke(prompt, fallback="[]", node_name="character_evolution")
        periods = safe_json_loads(content)
        if not isinstance(periods, list) or not periods:
            evolution[name] = [{"period_name": "全程", "source_chunks": source_chunks, **ent}]
        else:
            periods_list = []
            for p in periods:
                if not isinstance(p, dict):
                    continue
                attr = p.get("attributes", {}) or {}
                chunk_range = p.get("chunk_range", [min(source_chunks), max(source_chunks)])
                periods_list.append({
                    "period_name": p.get("period_name", "时期"),
                    "chunk_range": chunk_range,
                    "source_chunks": source_chunks,
                    "name": name,
                    "aliases": ent.get("aliases", []),
                    "gender": attr.get("gender", ent.get("gender", "unknown")),
                    "age": attr.get("age", ent.get("age", "")),
                    "personality": attr.get("personality", ent.get("personality", "")),
                    "alignment": attr.get("alignment", ent.get("alignment", "neutral")),
                    "appearance": ent.get("appearance", ""),
                    "background": ent.get("background", ""),
                    "emotion": ent.get("emotion", ""),
                    "timeline": ent.get("timeline", []),
                })
            evolution[name] = periods_list if periods_list else [{"period_name": "全程", "source_chunks": source_chunks, **ent}]

    state.character_evolution = evolution
    total_periods = sum(len(v) for v in evolution.values())
    logger.info(f"角色转变检测完成：{len(evolution)} 个角色 → {total_periods} 个时期")
    for name, periods in evolution.items():
        logger.info(f"  人物 {name}: {[p['period_name'] for p in periods]}")
    _record_step(state, "character_evolution")
    return state


def route_to_card_gens(state: AgentState):
    """将每个角色（或角色的每个时期）分发到角色卡生成 Worker"""
    if not state.character_evolution:
        return "qual_check"

    context_to_use = "\n\n".join(state.text_chunks[:3]) if state.text_chunks else ""

    sends = []
    card_id = 0
    for name, periods in state.character_evolution.items():
        for p in periods:
            card_id += 1
            card_state = CharacterCardState(
                agent_id=f"card_gen_{card_id}",
                character_info=json.dumps(p, ensure_ascii=False, indent=2),
                period=p.get("period_name", "全程"),
                context=context_to_use[:8000],
                question=state.original_input,
            )
            sends.append(Send("character_card_gen", card_state))

    logger.info(f"分发 {len(sends)} 个角色卡生成 Worker")
    return sends


def character_card_gen_node(state: CharacterCardState) -> dict:
    _ensure_llm()
    _worker_semaphore.acquire()
    try:
        prompt = CHARACTER_CARD_PROMPT.format(
            character_info=state.character_info,
            period=state.period,
            voice_rules=VOICE_MAPPING_RULES_TEXT,
        )
        logger.info(f"  🎴 {state.agent_id} 正在生成角色卡 ({state.period})...")
        start = time.time()

        content, tokens = invoke_with_retry(
            llm,
            prompt,
            max_retries=settings.llm_max_retries,
            base_delay=settings.llm_retry_base_delay,
            fallback="{}",
        )

        card = {}
        parsed_card = safe_json_loads(content)
        if isinstance(parsed_card, dict):
            card = parsed_card
        else:
            logger.warning(f"  {state.agent_id} 角色卡 JSON 解析失败")

        if card:
            card.setdefault("period", state.period)

        elapsed = time.time() - start
        logger.info(f"  ✅ {state.agent_id} 角色卡生成完成 ({elapsed:.1f}s)")
        return {"character_cards": {state.agent_id: card}} if card else {"character_cards": {}}
    finally:
        _worker_semaphore.release()


# ===== 质量自检 =====

def qual_check_node(state: AgentState) -> AgentState:
    """对角色卡进行质量评估。"""
    if state.direct_answer:
        state.quality_score = 1.0
        _record_step(state, "qual_check")
        return state

    if not state.character_cards:
        state.quality_score = 0.0
        logger.warning("未生成任何角色卡")
        _record_step(state, "qual_check")
        return state

    cards_list = list(state.character_cards.values())
    prompt = CHARACTER_QUALITY_PROMPT.format(
        question=state.original_input,
        character_cards=json.dumps(cards_list, ensure_ascii=False),
    )
    response, tokens = _invoke(prompt, fallback=json.dumps({"overall": 0.5}), node_name="qual_check")
    scores = safe_json_loads(response)
    if isinstance(scores, dict):
        state.quality_score = scores.get("overall", 0.5)
        state.quality_details = scores
    else:
        logger.warning(f"质量评估 JSON 解析失败: {response[:100]}")
        state.quality_score = 0.5

    _record_step(state, "qual_check", tokens)
    logger.info(f"角色卡质量评分: {state.quality_score:.2f}")
    return state


def aggregate_node(state: AgentState) -> AgentState:
    if not state.character_cards:
        logger.warning("无角色卡汇总")
    else:
        logger.info(f"汇总 {len(state.character_cards)} 张角色卡")
    _record_step(state, "aggregate")
    return state


# ===== 构建 Graph =====

def build_agent_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("entry", lambda s: s)
    workflow.add_node("file_reader", file_reader_node)
    workflow.add_node("direct_answer_node", direct_answer_node)
    workflow.add_node("character_extraction", character_extraction_node)
    workflow.add_node("character_merge", character_merge_node)
    workflow.add_node("character_evolution", character_evolution_node)
    workflow.add_node("character_card_gen", character_card_gen_node)
    workflow.add_node("aggregate", aggregate_node)
    workflow.add_node("qual_check", qual_check_node)

    workflow.set_entry_point("entry")

    workflow.add_edge("entry", "file_reader")

    # file_reader → 按段落块并行分发角色抽取 (route_after_file_reader 返回 Send)，失败则直接回答
    workflow.add_conditional_edges(
        "file_reader",
        route_after_file_reader,
        [["character_extraction"], "direct_answer_node"],
    )
    workflow.add_edge("direct_answer_node", "qual_check")

    # 角色抽取（Map 并行）→ 合并 → 演变检测
    workflow.add_edge("character_extraction", "character_merge")
    workflow.add_edge("character_merge", "character_evolution")

    # 演变检测 → 并行角色卡生成 → 汇总
    workflow.add_conditional_edges(
        "character_evolution",
        route_to_card_gens,
        [["character_card_gen"], "aggregate"],
    )
    workflow.add_edge("character_card_gen", "aggregate")

    workflow.add_edge("aggregate", "qual_check")

    workflow.add_edge("qual_check", END)

    return workflow.compile()