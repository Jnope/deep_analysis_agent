"""edge-tts 语音合成引擎。

将角色卡中的 voice_params 映射为 edge-tts 的预置声线 + pitch/rate/volume 参数，
生成角色样音音频文件。
"""

import asyncio
import json
import os
from typing import List, Optional, Tuple

import edge_tts
from loguru import logger


# (voice_type, tone) → edge-tts 预置声线（中文）
# 用二维组合扩展可选声线，使同 voice_type 不同 tone 的角色也能区分
VOICE_TONE_MAP = {
    # 男-年轻
    ("male_young", "bright"):  "zh-CN-YunxiNeural",
    ("male_young", "warm"):    "zh-CN-YunxiNeural",
    ("male_young", "neutral"): "zh-CN-YunyangNeural",
    ("male_young", "cold"):    "zh-CN-YunzeNeural",
    ("male_young", "dark"):    "zh-CN-YunzeNeural",
    # 男-低沉
    ("male_deep", "bright"):   "zh-CN-YunjianNeural",
    ("male_deep", "warm"):     "zh-CN-YunyangNeural",
    ("male_deep", "neutral"):  "zh-CN-YunjianNeural",
    ("male_deep", "cold"):     "zh-CN-YunzeNeural",
    ("male_deep", "dark"):     "zh-CN-YunjianNeural",
    # 女-温柔
    ("female_gentle", "bright"): "zh-CN-XiaoyiNeural",
    ("female_gentle", "warm"):   "zh-CN-XiaoxiaoNeural",
    ("female_gentle", "neutral"): "zh-CN-XiaochenNeural",
    ("female_gentle", "cold"):   "zh-CN-XiaomoNeural",
    ("female_gentle", "dark"):   "zh-CN-XiaoruiNeural",
    # 女-清脆
    ("female_strong", "bright"): "zh-CN-XiaozhenNeural",
    ("female_strong", "warm"):   "zh-CN-XiaoyanNeural",
    ("female_strong", "neutral"): "zh-CN-XiaoxuanNeural",
    ("female_strong", "cold"):   "zh-CN-XiaomoNeural",
    ("female_strong", "dark"):   "zh-CN-XiaozhenNeural",
    # 童声
    ("child", "bright"):  "zh-CN-XiaoshuangNeural",
    ("child", "warm"):    "zh-CN-XiaoshuangNeural",
    ("child", "neutral"): "zh-CN-YunxiaNeural",
    ("child", "cold"):    "zh-CN-YunxiaNeural",
    ("child", "dark"):    "zh-CN-YunxiaNeural",
    # 老年
    ("elderly", "bright"):  "zh-CN-YunzeNeural",
    ("elderly", "warm"):    "zh-CN-YunzeNeural",
    ("elderly", "neutral"): "zh-CN-YunzeNeural",
    ("elderly", "cold"):    "zh-CN-YunzeNeural",
    ("elderly", "dark"):    "zh-CN-YunzeNeural",
}

# 旁白专用声线（固定使用与角色不同的声线 + 降速）
NARRATOR_VOICE = "zh-CN-YunyangNeural"
NARRATOR_RATE = "-15%"
NARRATOR_PITCH = "-3Hz"

DEFAULT_SAMPLE_TEXT = "你好，我是这个角色。我的性格独特，经历丰富，在故事中扮演着重要的角色。"


def _voice_params_to_edge(card: dict, is_narrator: bool = False) -> Tuple[str, str, str, str]:
    """将角色卡的 voice_params 映射为 edge-tts 参数。

    根据 importance 分级调整区分度：
    - protagonist: 声线选择更有辨识度，pitch/rate 偏移放大 3 倍
    - supporting: 正常偏移，放大 2 倍
    - minor: 偏移压缩到 0.5 倍，减少配角之间的差异

    Returns:
        (voice, pitch, rate, volume)
    """
    vp = card.get("voice_params", {})
    voice_type = vp.get("voice_type", "male_deep")
    tone = vp.get("tone", "neutral")
    importance = card.get("importance", "minor")

    if is_narrator:
        voice = NARRATOR_VOICE
        pitch = NARRATOR_PITCH
        rate = NARRATOR_RATE
    else:
        voice = VOICE_TONE_MAP.get((voice_type, tone), "zh-CN-YunjianNeural")
        ssml_pitch = vp.get("ssml_pitch", "+0%")
        ssml_rate = vp.get("ssml_rate", "+0%")

        if importance == "protagonist":
            pitch = _percent_to_hz(ssml_pitch, amplify=3)
            rate = _amplify_rate(ssml_rate, amplify=3)
        elif importance == "supporting":
            pitch = _percent_to_hz(ssml_pitch, amplify=2)
            rate = _amplify_rate(ssml_rate, amplify=2)
        else:
            pitch = _percent_to_hz(ssml_pitch, amplify=1)
            rate = _amplify_rate(ssml_rate, amplify=1)

    # volume: energy 50 → +0%, energy 80 → +30%, energy 20 → -30%
    energy = vp.get("energy", 50)
    vol_offset = (energy - 50) * 2
    if importance == "minor":
        vol_offset = int(vol_offset * 0.5)
    elif importance == "protagonist":
        vol_offset = int(vol_offset * 1.5)
    if vol_offset > 0:
        volume = f"+{vol_offset}%"
    elif vol_offset < 0:
        volume = f"{vol_offset}%"
    else:
        volume = "+0%"

    return voice, pitch, rate, volume


def _percent_to_hz(pitch_str: str, amplify: int = 1) -> str:
    """将百分比 pitch（如 +5%、-10%）转换为 Hz 格式（如 +10Hz、-20Hz）。

    amplify 倍数放大偏移量，使音高差异更明显。
    """
    try:
        s = pitch_str.strip().replace("%", "")
        if s.startswith("+"):
            val = int(s[1:])
        elif s.startswith("-"):
            val = int(s)
        else:
            val = int(s)
        val *= amplify
        if val > 0:
            return f"+{val}Hz"
        elif val < 0:
            return f"{val}Hz"
        return "+0Hz"
    except (ValueError, TypeError):
        return "+0Hz"


def _amplify_rate(rate_str: str, amplify: int = 1) -> str:
    """放大 rate 偏移量，如 +10% amplify=2 → +20%。"""
    try:
        s = rate_str.strip().replace("%", "")
        if s.startswith("+"):
            val = int(s[1:])
        elif s.startswith("-"):
            val = int(s)
        else:
            val = int(s)
        val *= amplify
        if val > 0:
            return f"+{val}%"
        elif val < 0:
            return f"{val}%"
        return "+0%"
    except (ValueError, TypeError):
        return "+0%"


async def _generate_one(card: dict, text: str, output_path: str) -> str:
    """为单个角色卡生成一段样音。"""
    voice, pitch, rate, volume = _voice_params_to_edge(card)
    name = card.get("name", "unknown")

    communicate = edge_tts.Communicate(text, voice, pitch=pitch, rate=rate, volume=volume)
    await communicate.save(output_path)
    logger.info(f"  🎵 {name} → {output_path} (voice={voice}, pitch={pitch}, rate={rate}, volume={volume})")
    return output_path


def generate_sample_audio(
    character_cards: List[dict],
    output_dir: str = "./output/audio",
    sample_text: Optional[str] = None,
) -> List[str]:
    """为角色卡列表生成样音音频文件。

    Args:
        character_cards: 角色卡列表（dict 列表）
        output_dir: 输出目录
        sample_text: 样音文本，默认使用通用测试句

    Returns:
        生成的音频文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    text = sample_text or DEFAULT_SAMPLE_TEXT

    results = []
    for card in character_cards:
        name = card.get("name", "unknown")
        period = card.get("period", "全程")
        safe_name = name.replace("/", "_").replace(" ", "_")
        filename = f"{safe_name}_{period}.mp3"
        output_path = os.path.join(output_dir, filename)

        try:
            asyncio.run(_generate_one(card, text, output_path))
            results.append(output_path)
        except Exception as e:
            logger.error(f"  ❌ {name} 样音生成失败: {e}")

    logger.info(f"样音生成完成：{len(results)}/{len(character_cards)} 成功，输出目录: {output_dir}")
    return results


def generate_sample_audio_from_json(
    json_path: str,
    output_dir: str = "./output/audio",
    sample_text: Optional[str] = None,
) -> List[str]:
    """从角色卡 JSON 文件生成样音。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = list(data.values())
    if not isinstance(data, list):
        data = [data]

    return generate_sample_audio(data, output_dir, sample_text)


# ===== 旁白角色卡生成 =====

def generate_narrator_card(
    character_cards: List[dict],
    sample_text: str,
    llm=None,
) -> Tuple[Optional[dict], int]:
    """用 LLM 生成旁白角色卡。

    Args:
        character_cards: 已有的角色卡列表（供 LLM 参考小说风格）
        sample_text: 小说片段（让 LLM 感受叙事风格）
        llm: 可选的 LLM 实例，若不提供则使用全局

    Returns:
        旁白角色卡 dict，含 voice_params
    """
    from src.agents.prompts import NARRATOR_CARD_PROMPT
    from src.core.state import NarratorCard
    from src.utils.llm_utils import structured_extract, create_llm

    if llm is None:
        llm = create_llm()

    prompt = NARRATOR_CARD_PROMPT.format(
        character_cards=json.dumps(character_cards[:5], ensure_ascii=False),
        sample_text=sample_text[:2000],
    )
    result, tokens = structured_extract(llm, prompt, NarratorCard, max_retries=1)
    if result is not None:
        return result.model_dump(), tokens
    logger.warning("旁白角色卡生成失败")
    return None, tokens


# ===== 旁白/对话分拣 =====

def _chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    """将文本按段落和字符数切分为适合 LLM 处理的小块。"""
    paragraphs = text.split("\n")
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 > chunk_size and current:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def _markers_to_segments(chunk: str, markers: List) -> List[dict]:
    """将 LLM 返回的分拣标记（起止位置）转换为包含原文的片段列表。

    后端按 [start, end) 切分原文生成 text，避免 LLM 重复输出原文。
    """
    total_len = len(chunk)
    segments: List[dict] = []
    for m in markers:
        start = max(0, m.start)
        end = min(total_len, m.end)
        if end <= start:
            continue
        segments.append({
            "speaker": m.speaker,
            "text": chunk[start:end],
            "emotion": m.emotion,
        })
    return segments


def segment_dialogue(
    text: str,
    characters: List[dict],
    llm=None,
    chunk_size: int = 1500,
) -> List[dict]:
    """用 LLM 将文本段落分拣为旁白/角色对话片段。

    支持分块处理长文本：将文本切分为小块逐段分拣，最后合并结果。

    LLM 只返回每个片段的起止位置（markers），原文由后端按位置切分，
    避免 LLM 重复输出原文导致 token 翻倍/超长。

    Args:
        text: 小说文本段落
        characters: 角色卡列表（含 name 和 aliases）
        llm: 可选的 LLM 实例
        chunk_size: 每块最大字符数，默认 1500

    Returns:
        [{"speaker": "旁白"或角色名, "text": "...", "emotion": "..."}]
    """
    from src.agents.prompts import DIALOGUE_SEGMENTATION_PROMPT
    from src.core.state import SegmentationMarkerResult
    from src.utils.llm_utils import structured_extract, create_llm

    if llm is None:
        llm = create_llm()

    char_names = []
    for c in characters:
        names = [c.get("name", "")]
        names.extend(c.get("aliases", []))
        char_names.append({"name": c.get("name", ""), "aliases": c.get("aliases", [])})

    chunks = _chunk_text(text, chunk_size)
    all_segments: List[dict] = []
    total = len(chunks)

    for i, chunk in enumerate(chunks, 1):
        logger.info(f"  分拣第 {i}/{total} 块（{len(chunk)} 字符）...")
        prompt = DIALOGUE_SEGMENTATION_PROMPT.format(
            characters=json.dumps(char_names, ensure_ascii=False),
            text=chunk,
        )
        result, _ = structured_extract(llm, prompt, SegmentationMarkerResult, max_retries=1)
        if result is not None and result.markers:
            all_segments.extend(_markers_to_segments(chunk, result.markers))
        else:
            logger.warning(f"第 {i} 块分拣失败，该块按旁白处理")
            all_segments.append({"speaker": "旁白", "text": chunk, "emotion": "平静"})

    if not all_segments:
        logger.warning("对话分拣完全失败，整段按旁白处理")
        return [{"speaker": "旁白", "text": text, "emotion": "平静"}]

    return all_segments


# ===== 多角色朗读合成 =====

async def _generate_segment_audio(
    speaker: str,
    text: str,
    voice: str,
    pitch: str,
    rate: str,
    volume: str,
    output_path: str,
) -> str:
    """为单个朗读片段生成音频。"""
    communicate = edge_tts.Communicate(text, voice, pitch=pitch, rate=rate, volume=volume)
    await communicate.save(output_path)
    return output_path


def generate_narrated_audio(
    segments: List[dict],
    character_cards: List[dict],
    narrator_card: Optional[dict] = None,
    output_dir: str = "./output/audio",
    output_filename: str = "narrated.mp3",
) -> str:
    """将分拣后的片段按各自声线合成音频，拼接为完整朗读文件。

    Args:
        segments: 分拣结果 [{"speaker", "text", "emotion"}]
        character_cards: 角色卡列表（含 voice_params）
        narrator_card: 旁白角色卡（含 voice_params），若为 None 则用默认旁白声线

    Returns:
        最终拼接的音频文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # 构建声线映射：speaker_name → edge-tts 参数
    voice_map = {}
    for card in character_cards:
        name = card.get("name", "")
        if name:
            voice_map[name] = _voice_params_to_edge(card)
            for alias in card.get("aliases", []):
                voice_map[alias] = voice_map[name]

    # 旁白声线：使用专用旁白声线，明显降速 + 降音高，与角色拉开差距
    if narrator_card:
        voice_map["旁白"] = _voice_params_to_edge(narrator_card, is_narrator=True)
    else:
        voice_map["旁白"] = (NARRATOR_VOICE, NARRATOR_PITCH, NARRATOR_RATE, "+0%")

    # 逐片段生成临时音频
    temp_files = []
    for i, seg in enumerate(segments):
        speaker = seg.get("speaker", "旁白")
        text = seg.get("text", "").strip()
        if not text:
            continue

        voice, pitch, rate, volume = voice_map.get(speaker, voice_map["旁白"])
        temp_path = os.path.join(output_dir, f"_seg_{i:04d}.mp3")

        try:
            asyncio.run(_generate_segment_audio(speaker, text, voice, pitch, rate, volume, temp_path))
            temp_files.append(temp_path)
            logger.info(f"  [{i+1}/{len(segments)}] {speaker}: {text[:30]}... → {temp_path}")
        except Exception as e:
            logger.error(f"  [{i+1}] {speaker} 音频生成失败: {e}")

    # 拼接所有片段
    if not temp_files:
        logger.error("无音频片段可拼接")
        return ""

    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "wb") as out_f:
        for tf in temp_files:
            with open(tf, "rb") as in_f:
                out_f.write(in_f.read())

    # 清理临时文件
    for tf in temp_files:
        os.unlink(tf)

    logger.info(f"朗读音频合成完成: {output_path} ({len(temp_files)} 段)")
    return output_path
