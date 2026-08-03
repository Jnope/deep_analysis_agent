# ===== 小说角色卡生成 Agent 专用 Prompt =====

FILE_PATH_EXTRACTION_PROMPT = """
从用户的问题中提取所有本地小说文件路径或目录路径。

规则：
1. 提取所有看起来像文件路径的字符串（绝对路径、相对路径、带扩展名的文件名）
2. 路径可能包含空格，用引号包裹的路径需要去掉引号
3. 如果用户提到目录（如 ./docs/ 或 /home/user/reports），也提取
4. 如果没有文件路径，返回空数组

仅输出JSON，不要有其他内容：
{{
    "paths": ["/path/to/novel.txt"]
}}

用户问题：{question}
"""

CHARACTER_EXTRACTION_PROMPT = """
你是小说角色分析专家。请从以下文本片段中提取所有出现的角色，并记录他们的属性。

【文本片段】(第 {chunk_index} 块)
{text}

请提取每个角色的以下属性：
- name: 角色名
- aliases: 别名/绰号列表
- gender: male/female/unknown
- age: 年龄或年龄段（少年/青年/中年/老年/具体年龄）
- personality: 性格特征关键词（3-5个）
- alignment: good/evil/neutral
- appearance: 外貌描述（如有）
- background: 身份背景（如有）
- emotion: 本片段中的主要情绪

输出JSON数组，每个角色一个对象：
[
    {{"name": "角色名", "aliases": ["别名"], "gender": "male", "age": "青年", "personality": "坚韧、果断", "alignment": "good", "appearance": "...", "background": "...", "emotion": "..."}},
    ...
]

如果没有找到任何角色，输出空数组 []。
"""

CHARACTER_EVOLUTION_PROMPT = """
你是角色转变分析专家。以下是角色"{name}"在小说中的合并信息：

【角色信息】
{entity_info}

【出现位置（chunk索引）】
{source_chunks}

其中 timeline 字段按时间顺序记录了该角色在不同文本块中的属性快照，每条包含 chunk_index 和当时的 age、personality、alignment、emotion。

请根据 timeline 中的属性变化轨迹，判断该角色是否经历了显著转变。
判断标准（满足任一即算显著转变）：
1. 性格关键词变化超过3个
2. 阵营反转（good↔evil）
3. 年龄跨度过大（如少年→中年→老年）
4. 身份/地位剧变（如平民→帝王）

若存在显著转变，请按时期划分，输出每个时期的属性快照（从对应 timeline 条目中取值）：
[
    {{"period_name": "前期", "chunk_range": [0, 15], "attributes": {{"age": "...", "personality": "...", "alignment": "..."}}}},
    {{"period_name": "后期", "chunk_range": [16, 40], "attributes": {{"age": "...", "personality": "...", "alignment": "..."}}}}
]

若不存在显著转变，输出空数组 []。
"""

CHARACTER_CARD_PROMPT = """
你是角色卡设计专家。基于以下角色信息，生成完整的角色卡和音色参数。

角色信息：{character_info}
时期：{period}

请输出JSON（仅输出JSON，不要其他内容）：
{{
    "name": "角色名",
    "aliases": ["别名"],
    "gender": "male/female",
    "age": "年龄",
    "personality": "性格描述",
    "alignment": "good/evil/neutral",
    "appearance": "外貌描述",
    "background": "背景简介",
    "period": "时期标识",
    "voice_params": {{
        "pitch": 50,
        "speaking_rate": 50,
        "energy": 50,
        "tone": "warm",
        "voice_type": "male_deep",
        "description": "自然语言音色描述",
        "ssml_style": "cheerful",
        "ssml_pitch": "+5%",
        "ssml_rate": "+0%"
    }}
}}

{voice_rules}
"""

CHARACTER_QUALITY_PROMPT = """
你是一个严格的质量评估员。请评估以下角色卡生成结果的质量。

【原始需求】
{question}

【角色卡列表】
{character_cards}

请从以下维度打分（0-1分），仅输出JSON：
1. 完整性：是否覆盖了小说中的所有主要角色？
2. 准确性：角色属性是否准确，音色参数是否合理？
3. 一致性：同一角色不同时期的属性是否连贯？

{{
    "completeness": 0.9,
    "accuracy": 0.8,
    "consistency": 0.85,
    "overall": 0.85
}}
"""