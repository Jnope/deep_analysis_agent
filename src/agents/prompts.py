# ===== 小说角色卡生成 Agent 专用 Prompt =====

FILE_PATH_EXTRACTION_PROMPT = """
从用户的问题中提取所有本地小说文件路径或目录路径。

规则：
1. 提取所有看起来像文件路径的字符串（绝对路径、相对路径、带扩展名的文件名）
2. 路径可能包含空格，用引号包裹的路径需要去掉引号
3. 如果用户提到目录（如 ./docs/ 或 /home/user/reports），也提取
4. 如果没有文件路径，paths 输出空数组

用户问题：{question}
"""

CHARACTER_EXTRACTION_PROMPT = """
你是小说角色分析专家。请从以下文本片段中提取所有出现的角色，并记录他们的属性。

【文本片段】(第 {chunk_index} 块)
{text}

如果没有找到任何角色，输出空数组 items=[]。
"""

ALIAS_CLEANUP_PROMPT = """
你是角色别名清洗专家。以下是从小说中提取的所有角色及其别名列表。请逐个角色筛选，只保留**专属于该角色的独特称呼**（如绰号、化名、江湖名号、亲密昵称）。

必须剔除的别名类型：
1. 通用称呼：先生、小姐、公子、兄台、前辈、阁下、大人、姑娘、大叔、大娘、师父、师兄、师姐、道友、掌柜、掌门、殿下、陛下等
2. 人称代词/泛指：本才子、本座、在下、鄙人、小可、老夫、此人、那位、这厮、那厮等
3. 别人的名字：如果某个别名实际是另一个独立角色的名字，则剔除
4. 描述性词语而非称呼：如"辣货""漂亮小妞""死人妖"等纯描述

输入的角色列表：
{characters}

请为每个角色只保留专属称呼，剔除上述无效别名。输出 items 数组，每个元素包含 name 和 aliases。

如果一个角色的别名应全部剔除，aliases 输出空数组。
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

若存在显著转变，请按时期划分，输出每个时期的属性快照（periods，从对应 timeline 条目中取值）：
每个时期包含 period_name、chunk_range（[起始, 结束]）、attributes（含 age/personality/alignment）。

若不存在显著转变，periods 输出空数组。
"""

CHARACTER_CARD_PROMPT = """
你是角色卡设计专家。基于以下角色信息，生成完整的角色卡和音色参数。

角色信息：{character_info}
时期：{period}

请填充所有字段：name、aliases、gender、age、personality、alignment、appearance、background、period、voice_params。
其中 voice_params 包含 pitch/speaking_rate/energy（0-100）、tone/voice_type/ssml_style、description、ssml_pitch/ssml_rate。

{voice_rules}
"""

CHARACTER_QUALITY_PROMPT = """
你是一个严格的质量评估员。请评估以下角色卡生成结果的质量。

【原始需求】
{question}

【角色卡列表】
{character_cards}

请从以下维度打分（0-1分）：completeness（完整性，是否覆盖小说所有主要角色）、accuracy（准确性，角色属性是否准确、音色是否合理）、consistency（一致性，同一角色不同时期是否连贯）。综合分取 overall。
"""