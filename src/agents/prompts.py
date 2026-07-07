INTENT_RECOGNITION_PROMPT = """
你是一个意图识别专家。根据用户的问题，判断属于以下哪种类型：

1. direct_answer：简单问答类，不需要查阅外部文档或上下文，直接回答即可。
   例如："今天天气如何"、"1+1等于几"、"什么是机器学习"

2. tool_call：需要调用外部工具或API才能回答的问题。
   例如："帮我查一下股票价格"、"搜索最新的AI新闻"

3. file_analysis：用户提到了本地文件路径或目录，需要读取并分析文件内容。
   例如："分析 /home/user/docs/architecture.pdf 的高可用性"
   例如："读取 ./report/ 下的所有文档并总结"

4. deep_analysis：需要对长文本/文档进行深入分析，可能需要压缩、拆解、多角度评估。
   例如：分析50页的架构文档并指出风险点、总结一本书的核心观点、对比两份合同条款的差异

请只输出一个单词：direct_answer / tool_call / file_analysis / deep_analysis

用户问题：{question}
"""

FILE_PATH_EXTRACTION_PROMPT = """
从用户的问题中提取所有本地文件路径或目录路径。

规则：
1. 提取所有看起来像文件路径的字符串（绝对路径、相对路径、带扩展名的文件名）
2. 路径可能包含空格，用引号包裹的路径需要去掉引号
3. 如果用户提到目录（如 ./docs/ 或 /home/user/reports），也提取
4. 如果没有文件路径，返回空数组

仅输出JSON，不要有其他内容：
{{
    "paths": ["/path/to/file1.pdf", "./docs/"]
}}

用户问题：{question}
"""

DIRECT_ANSWER_PROMPT = """
请直接回答以下问题，保持简洁准确。

问题：{question}
"""

DEEP_ANALYSIS_SUPERVISOR_PROMPT = """
你是一个任务分解专家。根据以下上下文和用户需求，制定分析计划。

你需要做两件事：
1. 确定本次分析需要提取的实体类型和属性（entity_schema）
2. 为每个实体的深度分析指定专业分析师角色

【上下文信息】
{context}

【用户需求】
{question}

请输出JSON格式的分析计划：
{{
    "entity_schema": {{
        "entity_type": "实体类型（如：人物/系统组件/合同条款/事件）",
        "attributes": ["属性1（如：性格）", "属性2（如：经历）", "属性3（如：关系）"]
    }},
    "tasks": [
        {{"id": 1, "description": "分析维度描述", "assigned_to": "analyst_1", "role": "你是一个...专家，擅长..."}},
        ...
    ]
}}
"""

ENTITY_EXTRACTION_PROMPT = """
你是一个信息抽取专家。请从以下文本片段中，按照给定的实体类型和属性，提取所有出现的实体。

【实体类型】
{entity_type}

【需要提取的属性】
{attributes}

【文本片段】
{text}

请输出JSON数组，每个实体一个对象。属性值为文本中直接体现的内容，如果文本中没有体现则留空：
[
    {{"name": "实体名称", "属性1": "值", "属性2": "值", ...}},
    ...
]

如果没有找到任何实体，输出空数组 []。
"""

ENTITY_MERGE_PROMPT = """
你是一个实体合并专家。以下是从不同文本片段中提取的实体信息，可能存在重复、别名或矛盾。

【实体类型】
{entity_type}

【待合并的实体列表】
{entities}

请执行以下操作：
1. 合并同义实体（同一实体的不同称呼，如"张三"和"老张"应合并）
2. 合并同一实体的多个属性片段，取最完整的信息
3. 如果存在矛盾属性（如性格前后变化），保留所有版本并标注来源片段

输出合并后的JSON数组：
[
    {{"name": "实体名称", "aliases": ["别名1", "别名2"], "属性1": "合并后的值", "属性2": "值", ...}},
    ...
]
"""

DEEP_ANALYSIS_REDUCE_PROMPT = """
{role}

请基于以下信息，对该实体进行深度分析。

【实体信息】
{entity_info}

【上下文】
{context}

【用户需求】
{question}

请输出详细的分析结论：
"""

DEEP_ANALYSIS_WORKER_PROMPT = """
{role}

请根据以下上下文，从你专业的角度完成分析任务。输出详细的分析结论。

【上下文】
{context}

【分析任务】
{task_description}

请输出分析结论：
"""

TOOL_CALL_QUALITY_PROMPT = """
你是一个严格的质量评估员。请评估以下工具调用执行结果的质量。

【用户需求】
{question}

【执行结果】
{results}

请从以下维度打分（0-1分），仅输出JSON：
1. 正确性：结果是否正确回答了用户问题？
2. 规范性：工具调用方式是否合理？
3. 完整性：是否提供了足够的细节？

{{
    "correctness": 0.9,
    "standardization": 0.8,
    "completeness": 0.85,
    "overall": 0.85
}}
"""

DEEP_ANALYSIS_QUALITY_PROMPT = """
你是一个严格的质量评估员。请评估以下长文本分析结果的质量。

【原始需求】
{question}

【分析任务】
{tasks}

【分析结果】
{results}

请从以下维度打分（0-1分），仅输出JSON：
1. 完整性：是否覆盖了所有要求的分析维度？
2. 准确性：结论是否有依据，是否存在事实错误？
3. 逻辑性：分析推理是否连贯？

{{
    "completeness": 0.9,
    "accuracy": 0.8,
    "logic": 0.85,
    "overall": 0.85
}}
"""