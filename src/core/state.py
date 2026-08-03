from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field


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

    # ===== 质量评估 =====
    quality_score: Optional[float] = None
    quality_details: Dict[str, float] = Field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3

    # ===== 可观测性 =====
    total_tokens_used: int = 0
    total_cost: float = 0.0
    execution_steps: Annotated[List[str], _merge_list] = Field(default_factory=list)