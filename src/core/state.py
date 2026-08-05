from typing import List, Dict, Optional, Annotated, Literal
from pydantic import BaseModel, Field, field_validator


class VoiceParams(BaseModel):
    """角色音色参数，三层格式：通用数值 + 自然语言 + SSML"""
    pitch: int = Field(default=50, ge=0, le=100, description="音高，50为中性")
    speaking_rate: int = Field(default=50, ge=0, le=100, description="语速，50为正常")
    energy: int = Field(default=50, ge=0, le=100, description="能量/力度，50为中等")
    tone: Literal["warm", "cold", "bright", "dark", "neutral"] = Field(default="neutral", description="语气基调")
    voice_type: Literal[
        "male_deep", "male_young", "female_gentle", "female_strong", "child", "elderly"
    ] = Field(default="male_deep", description="音色类型")
    description: str = Field(default="", description="自然语言音色描述")
    ssml_style: Literal["cheerful", "sad", "angry", "calm", "serious"] = Field(default="calm", description="SSML 情绪风格")
    ssml_pitch: str = Field(default="+0%", description="SSML 音高，如 +5%/-10%")
    ssml_rate: str = Field(default="+0%", description="SSML 语速，如 +10%/-10%")

    @field_validator("pitch", "speaking_rate", "energy", mode="before")
    @classmethod
    def _coerce_int(cls, v):
        try:
            v = int(v)
            return max(0, min(100, v))
        except (TypeError, ValueError):
            return 50

    @field_validator("tone", mode="before")
    @classmethod
    def _coerce_tone(cls, v):
        valid = {"warm", "cold", "bright", "dark", "neutral"}
        return v if v in valid else "neutral"

    @field_validator("voice_type", mode="before")
    @classmethod
    def _coerce_voice_type(cls, v):
        valid = {"male_deep", "male_young", "female_gentle", "female_strong", "child", "elderly"}
        return v if v in valid else "male_deep"

    @field_validator("ssml_style", mode="before")
    @classmethod
    def _coerce_ssml_style(cls, v):
        valid = {"cheerful", "sad", "angry", "calm", "serious"}
        return v if v in valid else "calm"

    @field_validator("ssml_pitch", "ssml_rate", "description", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        return str(v) if v is not None else ""


class CharacterCard(BaseModel):
    """结构化角色卡"""
    name: str = Field(default="", description="角色名")
    aliases: List[str] = Field(default_factory=list, description="专属别名/绰号")
    gender: Literal["male", "female", "unknown"] = Field(default="unknown", description="性别")
    age: str = Field(default="", description="年龄或年龄段")
    personality: str = Field(default="", description="性格描述")
    alignment: Literal["good", "evil", "neutral"] = Field(default="neutral", description="阵营")
    appearance: str = Field(default="", description="外貌描述")
    background: str = Field(default="", description="身份背景")
    period: str = Field(default="全程", description="时期标识")
    importance: Literal["protagonist", "supporting", "minor"] = Field(default="minor", description="角色重要程度：protagonist=主角(男一/女一), supporting=重要配角(二号等), minor=普通配角")
    voice_params: VoiceParams = Field(default_factory=VoiceParams, description="音色参数")

    @field_validator("personality", "age", "appearance", "background", "period", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        if isinstance(v, list):
            return "、".join(str(x).strip() for x in v if str(x).strip())
        return str(v) if v is not None else ""

    @field_validator("gender", mode="before")
    @classmethod
    def _coerce_gender(cls, v):
        if isinstance(v, str) and v.lower() in ("male", "female", "unknown"):
            return v.lower()
        return "unknown"

    @field_validator("alignment", mode="before")
    @classmethod
    def _coerce_alignment(cls, v):
        if isinstance(v, str) and v.lower() in ("good", "evil", "neutral"):
            return v.lower()
        return "neutral"

    @field_validator("aliases", mode="before")
    @classmethod
    def _coerce_aliases(cls, v):
        if isinstance(v, str):
            return [v]
        return v or []

    @field_validator("voice_params", mode="before")
    @classmethod
    def _coerce_voice_params(cls, v):
        if isinstance(v, dict):
            return VoiceParams(**v)
        return v

    @field_validator("importance", mode="before")
    @classmethod
    def _coerce_importance(cls, v):
        valid = {"protagonist", "supporting", "minor"}
        if isinstance(v, str) and v.lower() in valid:
            return v.lower()
        return "minor"


class CharacterExtraction(BaseModel):
    """角色抽取的结构化输出"""
    name: str = Field(default="", description="角色名")
    aliases: List[str] = Field(default_factory=list, description="专属别名/绰号")
    gender: Literal["male", "female", "unknown"] = Field(default="unknown", description="性别")
    age: str = Field(default="", description="年龄或年龄段")
    personality: str = Field(default="", description="性格特征")
    alignment: Literal["good", "evil", "neutral"] = Field(default="neutral", description="阵营")
    appearance: str = Field(default="", description="外貌描述")
    background: str = Field(default="", description="身份背景")
    emotion: str = Field(default="", description="本片情绪")
    importance: Literal["protagonist", "supporting", "minor"] = Field(default="minor", description="角色重要程度：protagonist=主角(男一/女一), supporting=重要配角(二号等), minor=普通配角")

    @field_validator("personality", "emotion", "age", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        if isinstance(v, list):
            return "、".join(str(x).strip() for x in v if str(x).strip())
        return str(v) if v is not None else ""

    @field_validator("gender", mode="before")
    @classmethod
    def _coerce_gender(cls, v):
        if isinstance(v, str) and v.lower() in ("male", "female", "unknown"):
            return v.lower()
        return "unknown"

    @field_validator("alignment", mode="before")
    @classmethod
    def _coerce_alignment(cls, v):
        if isinstance(v, str) and v.lower() in ("good", "evil", "neutral"):
            return v.lower()
        return "neutral"

    @field_validator("aliases", mode="before")
    @classmethod
    def _coerce_aliases(cls, v):
        if isinstance(v, str):
            return [v]
        return v or []

    @field_validator("importance", mode="before")
    @classmethod
    def _coerce_importance(cls, v):
        valid = {"protagonist", "supporting", "minor"}
        if isinstance(v, str) and v.lower() in valid:
            return v.lower()
        return "minor"


class CharacterExtractionList(BaseModel):
    """角色抽取的列表包装（约束作用，实际内容按 items 数组）"""
    items: List[CharacterExtraction] = Field(default_factory=list, description="抽取到的角色列表")


class FilePathResult(BaseModel):
    """从用户输入中提取的小说文件路径"""
    paths: List[str] = Field(default_factory=list, description="找到的小说文件路径列表")

    @field_validator("paths", mode="before")
    @classmethod
    def _coerce_paths(cls, v):
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [p for p in v if isinstance(p, str) and p.strip()]
        return []


class AliasCleanupItem(BaseModel):
    """别名清洗后的单个角色"""
    name: str = Field(default="", description="角色名")
    aliases: List[str] = Field(default_factory=list, description="清洗后保留的专属别名")

    @field_validator("aliases", mode="before")
    @classmethod
    def _coerce_aliases(cls, v):
        if isinstance(v, str):
            return [v]
        return [a for a in (v or []) if isinstance(a, str)] if isinstance(v, list) else []


class AliasCleanupList(BaseModel):
    """别名清洗结果"""
    items: List[AliasCleanupItem] = Field(default_factory=list, description="清洗后的角色别名列表")


class EvolutionAttributes(BaseModel):
    """角色某时期的属性快照"""
    age: str = Field(default="", description="年龄或年龄段")
    personality: str = Field(default="", description="性格")
    alignment: Literal["good", "evil", "neutral"] = Field(default="neutral", description="阵营")

    @field_validator("age", "personality", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        if isinstance(v, list):
            return "、".join(str(x).strip() for x in v if str(x).strip())
        return str(v) if v is not None else ""

    @field_validator("alignment", mode="before")
    @classmethod
    def _coerce_alignment(cls, v):
        if isinstance(v, str) and v.lower() in ("good", "evil", "neutral"):
            return v.lower()
        return "neutral"


class EvolutionPeriod(BaseModel):
    """角色转变的一个时期"""
    period_name: str = Field(default="", description="时期名称")
    chunk_range: List[int] = Field(default_factory=list, description="chunk 范围 [起始, 结束]")
    attributes: EvolutionAttributes = Field(default_factory=EvolutionAttributes, description="该时期属性快照")

    @field_validator("chunk_range", mode="before")
    @classmethod
    def _coerce_range(cls, v):
        if isinstance(v, list):
            nums = [int(x) for x in v if isinstance(x, (int, float)) or str(x).isdigit()]
            return nums[:2]
        return []


class EvolutionResult(BaseModel):
    """角色转变检测结果"""
    periods: List[EvolutionPeriod] = Field(default_factory=list, description="划分的时期列表，空表示无显著转变")


class NarratorCard(BaseModel):
    """旁白角色卡"""
    name: str = Field(default="旁白", description="固定为旁白")
    period: str = Field(default="全程", description="时期")
    narrator_style: str = Field(default="", description="旁白风格描述")
    voice_params: VoiceParams = Field(default_factory=VoiceParams, description="旁白音色参数")


class Segment(BaseModel):
    """分拣后的单个朗读片段"""
    speaker: str = Field(default="旁白", description="说话者；角色名或'旁白'")
    text: str = Field(default="", description="原文")
    emotion: str = Field(default="平静", description="语气情绪")

    @field_validator("text", "emotion", "speaker", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        return str(v) if v is not None else ""


class SegmentationMarker(BaseModel):
    """对话/旁白分拣标记（只含切割位置，不含文本）。

    用于减少 LLM 输出量：LLM 只需返回每个片段的起止位置和说话者，
    对应原文由后端按 [start, end) 切分生成。
    """
    start: int = Field(default=0, ge=0, description="该片段在原文中的起始字符下标（含）")
    end: int = Field(default=0, ge=0, description="该片段在原文中的结束字符下标（不含）")
    speaker: str = Field(default="旁白", description="说话者；角色名或'旁白'")
    emotion: str = Field(default="平静", description="语气情绪")

    @field_validator("start", "end", mode="before")
    @classmethod
    def _coerce_pos(cls, v):
        try:
            return max(0, int(v))
        except (TypeError, ValueError):
            return 0

    @field_validator("emotion", "speaker", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        return str(v) if v is not None else ""

class SegmentationMarkerResult(BaseModel):
    """对话/旁白分拣标记结果（只含起止位置）"""
    markers: List[SegmentationMarker] = Field(default_factory=list, description="分拣标记列表，按 start 升序排列")


class QualityScore(BaseModel):
    """角色卡质量评估分"""
    completeness: float = Field(default=0.5, ge=0, le=1, description="完整性")
    accuracy: float = Field(default=0.5, ge=0, le=1, description="准确性")
    consistency: float = Field(default=0.5, ge=0, le=1, description="一致性")
    overall: float = Field(default=0.5, ge=0, le=1, description="综合分")

    @field_validator("completeness", "accuracy", "consistency", "overall", mode="before")
    @classmethod
    def _coerce_float(cls, v):
        try:
            v = float(v)
            return max(0.0, min(1.0, v))
        except (TypeError, ValueError):
            return 0.5


class EntityExtractionState(BaseModel):
    """角色抽取 Worker 的子状态，保留 chunk_index 记录时间线"""
    chunk_index: int = 0
    text: str = ""

    class Config:
        arbitrary_types_allowed = True


class CharacterCardState(BaseModel):
    """角色卡生成 Worker 的子状态"""
    agent_id: str = ""
    character_info: str = ""
    period: str = ""
    context: str = ""
    question: str = ""

    class Config:
        arbitrary_types_allowed = True


def _merge_dict(left: Dict, right: Dict) -> Dict:
    if left is None:
        left = {}
    if right is None:
        right = {}
    result = dict(left)
    result.update(right)
    return result


def _merge_list(left: List, right: List) -> List:
    if left is None:
        left = []
    if right is None:
        right = []
    if isinstance(right, list):
        return left + right
    return left + [right]


def _merge_int_add(left: Optional[int], right: Optional[int]) -> int:
    """整数值累加 reducer（用于并行 worker 累加 token）"""
    return int(left or 0) + int(right or 0)


class AgentState(BaseModel):
    """小说角色卡生成 Agent 的全局状态"""

    original_input: str = ""          # 用户需求（应包含小说文件路径）
    context_path: str = ""            # 小说文件路径

    extracted_file_paths: List[str] = Field(default_factory=list)
    text_chunks: Annotated[List[str], _merge_list] = Field(default_factory=list)  # 段落分块，保留顺序

    is_character_analysis: bool = False
    direct_answer: Optional[str] = None   # 文件读取失败时的提示

    extracted_entities: Annotated[List[dict], _merge_list] = Field(default_factory=list)  # 各块抽取的角色
    merged_entities: Optional[List[dict]] = None                                          # 合并去重后的角色
    character_evolution: Dict[str, List[dict]] = Field(default_factory=dict)              # 角色 → 时期列表
    character_cards: Annotated[Dict[str, dict], _merge_dict] = Field(default_factory=dict)  # agent_id → 角色卡
    narrator_card: Optional[dict] = Field(default=None, description="旁白角色卡")

    # ===== 质量评估 =====
    quality_score: Optional[float] = None
    quality_details: Dict[str, float] = Field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3

    # ===== 可观测性 =====
    total_tokens_used: Annotated[int, _merge_int_add] = 0
    total_cost: float = 0.0
    execution_steps: Annotated[List[str], _merge_list] = Field(default_factory=list)