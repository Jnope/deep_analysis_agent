"""音色参数映射工具。

根据角色属性（性别、年龄、性格、阵营）计算音色参数。
支持三层格式：通用数值参数 + 自然语言描述 + SSML 标记。
"""

from typing import Dict, Any

GENDER_AGE_MAP = {
    ("male", "少年"):    {"voice_type": "male_young",     "pitch": (50, 60)},
    ("male", "青年"):    {"voice_type": "male_deep",      "pitch": (35, 45)},
    ("male", "中年"):    {"voice_type": "male_deep",      "pitch": (30, 40)},
    ("male", "老年"):    {"voice_type": "elderly",        "pitch": (25, 35)},
    ("female", "少女"):  {"voice_type": "female_gentle",  "pitch": (60, 70)},
    ("female", "青年"):  {"voice_type": "female_strong",  "pitch": (55, 65)},
    ("female", "中年"):  {"voice_type": "female_gentle",  "pitch": (50, 60)},
    ("female", "老年"):  {"voice_type": "elderly",        "pitch": (45, 55)},
}

PERSONALITY_MAP = {
    "温柔": {"energy": (30, 40), "speaking_rate": (40, 45), "tone": "warm"},
    "暴躁": {"energy": (70, 80), "speaking_rate": (60, 70), "tone": "bright"},
    "阴沉": {"energy": (35, 45), "speaking_rate": (40, 50), "tone": "dark"},
    "活泼": {"energy": (65, 75), "speaking_rate": (60, 65), "tone": "bright"},
    "冷漠": {"energy": (30, 40), "speaking_rate": (45, 50), "tone": "cold"},
    "威严": {"energy": (55, 65), "speaking_rate": (40, 45), "tone": "dark"},
    "懦弱": {"energy": (25, 35), "speaking_rate": (45, 55), "tone": "warm"},
    "狡猾": {"energy": (40, 50), "speaking_rate": (50, 60), "tone": "cold"},
    "热血": {"energy": (65, 75), "speaking_rate": (55, 65), "tone": "bright"},
    "沉稳": {"energy": (40, 50), "speaking_rate": (40, 45), "tone": "neutral"},
    "傲慢": {"energy": (50, 60), "speaking_rate": (45, 50), "tone": "cold"},
    "善良": {"energy": (40, 50), "speaking_rate": (45, 55), "tone": "warm"},
}

ALIGNMENT_TONE_ADJUST = {
    "good":    {"tone_shift": "warm"},
    "evil":    {"tone_shift": "cold"},
    "neutral": {"tone_shift": None},
}

SSML_STYLE_MAP = {
    "warm": "cheerful",
    "cold": "calm",
    "bright": "cheerful",
    "dark": "serious",
    "neutral": "calm",
}

VOICE_TYPE_DESCRIPTION = {
    "male_young":     "清亮有活力的少年男性嗓音",
    "male_deep":      "低沉浑厚的成年男性嗓音",
    "female_gentle":  "温柔柔和的女性嗓音",
    "female_strong":  "清脆有力的女性嗓音",
    "child":          "稚嫩的童声",
    "elderly":        "苍老缓慢的老年嗓音",
}


def _midpoint(rng):
    return (rng[0] + rng[1]) // 2


def _to_ssml_pitch(pitch_val: int) -> str:
    offset = pitch_val - 50
    if offset > 0:
        return f"+{offset}%"
    elif offset < 0:
        return f"{offset}%"
    return "+0%"


def _to_ssml_rate(rate_val: int) -> str:
    offset = rate_val - 50
    if offset > 0:
        return f"+{offset * 2}%"
    elif offset < 0:
        return f"{offset * 2}%"
    return "+0%"


def _detect_age_group(age: str) -> str:
    age_lower = (age or "").lower()
    for keyword in ["少年", "少年", "孩童", "幼"]:
        if keyword in age_lower:
            return "少年"
    for keyword in ["青年", "年轻", "青年"]:
        if keyword in age_lower:
            return "青年"
    for keyword in ["中年", "壮年"]:
        if keyword in age_lower:
            return "中年"
    for keyword in ["老年", "老者", "暮年"]:
        if keyword in age_lower:
            return "老年"
    return "青年"


def _match_personality(personality: str) -> str:
    for keyword in PERSONALITY_MAP:
        if keyword in (personality or ""):
            return keyword
    return "沉稳"


def compute_voice_params(
    gender: str,
    age: str,
    personality: str,
    alignment: str,
) -> Dict[str, Any]:
    """根据角色属性计算音色参数。"""
    gender = (gender or "male").lower().strip()
    if gender not in ("male", "female"):
        gender = "male"

    age_group = _detect_age_group(age)
    base = GENDER_AGE_MAP.get((gender, age_group), GENDER_AGE_MAP[(gender, "青年")])

    personality_key = _match_personality(personality)
    pers = PERSONALITY_MAP[personality_key]

    align = ALIGNMENT_TONE_ADJUST.get((alignment or "neutral").lower(), ALIGNMENT_TONE_ADJUST["neutral"])
    tone = pers["tone"]
    if align["tone_shift"]:
        tone = align["tone_shift"]

    pitch = _midpoint(base["pitch"])
    speaking_rate = _midpoint(pers["speaking_rate"])
    energy = _midpoint(pers["energy"])

    voice_type = base["voice_type"]
    desc_base = VOICE_TYPE_DESCRIPTION.get(voice_type, "中性嗓音")

    rate_desc = "语速偏快" if speaking_rate > 55 else "语速偏慢" if speaking_rate < 45 else "语速适中"
    energy_desc = "充满力量" if energy > 60 else "轻柔低沉" if energy < 40 else "力度适中"
    description = f"{desc_base}，{rate_desc}，{energy_desc}"

    ssml_style = SSML_STYLE_MAP.get(tone, "calm")
    ssml_pitch = _to_ssml_pitch(pitch)
    ssml_rate = _to_ssml_rate(speaking_rate)

    return {
        "pitch": pitch,
        "speaking_rate": speaking_rate,
        "energy": energy,
        "tone": tone,
        "voice_type": voice_type,
        "description": description,
        "ssml_style": ssml_style,
        "ssml_pitch": ssml_pitch,
        "ssml_rate": ssml_rate,
    }


VOICE_MAPPING_RULES_TEXT = """
音色映射参考规则：
- 性别+年龄 → voice_type 和 pitch 基线：
  male + 少年 → male_young, pitch=50-60
  male + 青年 → male_deep, pitch=35-45
  male + 中年 → male_deep, pitch=30-40
  male + 老年 → elderly, pitch=25-35
  female + 少女 → female_gentle, pitch=60-70
  female + 青年 → female_strong, pitch=55-65
  female + 中年 → female_gentle, pitch=50-60
  female + 老年 → elderly, pitch=45-55
- 性格 → energy + speaking_rate + tone：
  温柔 → energy=30-40, rate=40-45, tone=warm
  暴躁 → energy=70-80, rate=60-70, tone=bright
  阴沉 → energy=35-45, rate=40-50, tone=dark
  活泼 → energy=65-75, rate=60-65, tone=bright
  冷漠 → energy=30-40, rate=45-50, tone=cold
  威严 → energy=55-65, rate=40-45, tone=dark
- 阵营 → tone 微调：
  good → tone 偏 warm/bright
  evil → tone 偏 cold/dark
  neutral → 不调整
- 重要程度(importance) → 音色区分度策略：
  protagonist(主角): pitch/rate/energy 偏移加大（偏离中性值更远），声线选择更有辨识度
  supporting(重要配角): 正常偏移，与主角有明显区分
  minor(普通配角): pitch/rate/energy 尽量接近中性（50附近），减少配角之间的差异
"""
