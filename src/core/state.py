from typing import List, Dict, Any, Optional, Literal, Annotated
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from enum import Enum
import operator


class CompressionStrategy(str, Enum):
    MAP_REDUCE = "map_reduce"
    STUFF = "stuff"
    EXTRACTIVE = "extractive"
    AUTO = "auto"

class IntentType(str, Enum):
    DIRECT_ANSWER = "direct_answer"
    TOOL_CALL = "tool_call"
    DEEP_ANALYSIS = "deep_analysis"
    FILE_ANALYSIS = "file_analysis"

class SubTask(BaseModel):
    id: int
    description: str
    assigned_to: str
    role: str = ""

class WorkerState(BaseModel):
    """单个 Worker 的子状态，用于并行分发"""
    agent_id: str = ""
    role: str = ""
    task_description: str = ""
    context: str = ""

    class Config:
        arbitrary_types_allowed = True

class TaskPlan(BaseModel):
    tasks: List[SubTask]

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
    return left + right

class AgentState(BaseModel):
    """多 Agent 系统的全局状态"""

    # ===== 基础消息 =====
    messages: List[BaseMessage] = Field(default_factory=list)
    original_input: str = ""

    # ===== 意图识别 =====
    intent: Optional[IntentType] = None
    direct_answer: Optional[str] = None
    extracted_file_paths: List[str] = Field(default_factory=list)

    # ===== 上下文压缩 =====
    context_path: str = ""  # 文件路径，优先于 raw_context
    raw_context: str = ""
    compressed_context: Optional[str] = None
    compression_ratio: float = 0.0
    compression_strategy: CompressionStrategy = CompressionStrategy.AUTO
    compression_quality: Optional[float] = None

    # ===== 任务编排 =====
    task_plan: Optional[dict] = None
    worker_results: Annotated[Dict[str, Any], _merge_dict] = Field(default_factory=dict)
    final_answer: Optional[str] = None

    # ===== Harness 评估 =====
    quality_score: Optional[float] = None
    quality_details: Annotated[Dict[str, float], _merge_dict] = Field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3

    # ===== 可观测性 =====
    total_tokens_used: int = 0
    total_cost: float = 0.0
    execution_steps: Annotated[List[str], _merge_list] = Field(default_factory=list)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    # ===== 记忆 =====
    short_term_memory: List[BaseMessage] = Field(default_factory=list)
    long_term_memory_context: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True