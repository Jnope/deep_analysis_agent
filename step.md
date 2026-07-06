# 实现步骤

## 整体架构流程

```
用户输入 (original_input)
    │
    ▼
[意图识别] ──→ 简单问答 ──→ 直接 LLM 回答 ──→ END
    │
    ├──→ 需工具调用 ──→ supervisor(调用工具) ──→ END
    │
    └──→ 长文本分析 ──→ compress ──→ supervisor(task拆解)
                                              │
                                              ▼
                                    多Agent并行分析
                                              │
                                              ▼
                                    质量自检 ←──┐
                                              │    不通过(重试)
                                              ▼
                                             END
```

## 步骤0：基础修复
- **pyproject.toml**：添加 `pydantic-settings`、`langchain-classic` 依赖
- **state.py**：补充 `SubTask` / `TaskPlan` 模型
- **compressor.py**：废弃 API → `langchain_classic`，补全 extractive 策略

## 步骤1：上下文压缩
原 `compressor.py` 已实现三种策略（map_reduce / stuff / extractive），仅需补全 extractive 并修复 import。

## 步骤2：意图识别 + 动态路由

### 2.1 新增 `intent_recognition_node`
接收用户的 `original_input`，调用 LLM 识别意图类型，返回 `intent`：
- `direct_answer`：简单问答，无需上下文
- `tool_call`：需要调用外部工具
- `deep_analysis`：长文本分析，需要完整压缩+多Agent流程

### 2.2 新增 `route_by_intent` 条件边
根据 `state.intent` 路由到不同节点：
- `direct_answer` → `direct_answer_node` → END
- `tool_call` → `supervisor_node` → 工具调用
- `deep_analysis` → `compress_context_node` → 完整流程

### 2.3 新增 `direct_answer_node`
直接调用 LLM 返回回答，跳过所有后续处理。

## 步骤3：任务拆解
`supervisor_node` 根据 `state.intent = deep_analysis` 时，按场景维度拆解子任务。
- Prompt 根据 `state.original_input` 动态适配
- 输出 JSON 存入 `state.task_plan`

## 步骤4：多Agent并行分析
`worker_agent_node` 接收 `state.task_plan.tasks`，单个 LLM 调用一次性分析所有子任务，返回汇总结果。
- 按子任务维度组织输出
- 结果存入 `state.worker_results`

## 步骤5：质量自检
`quality_harness_node` 根据 `state.intent` 选择评估标准：
- `deep_analysis`：完整性 + 准确性 + 逻辑性
- `tool_call`：结果正确性 + 工具调用规范性
评分低于阈值（默认 0.7）则 `should_retry` → supervisor。

## 步骤6：main.py 入口
加载 `.env` → 构建 graph → 传入文档内容 + 用户需求 → 流式/阻塞执行 → 输出最终结果。

## 步骤7：测试
- 编写测试用例：direct_answer、deep_analysis 两种场景
- 测试完整流程 end-to-end