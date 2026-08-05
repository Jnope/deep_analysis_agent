import argparse
import json
import os

from src.core.graph import build_agent_graph
from src.core.state import AgentState


def _safe_write_json(path: str, data) -> None:
    """安全写 JSON 文件，自动创建父目录。"""
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def cmd_generate(args):
    """生成角色卡"""
    if not args.file:
        args.file = input("请输入小说文件路径: ").strip()

    requirement = args.requirement or "分析这部小说中的所有角色，为每个（含转变的不同时期）生成角色卡（性格、好坏、性别、年龄等）及音色参数。"

    print(f"\n{'='*60}")
    print(f"小说文件: {args.file}")
    print(f"生成需求: {requirement}")
    print(f"{'='*60}\n")

    graph = build_agent_graph()
    initial_state = AgentState(
        original_input=f"分析小说文件 {args.file}。{requirement}",
        context_path=args.file,
        max_retries=args.max_retries,
    )
    final_state = graph.invoke(initial_state)

    print(f"\n{'='*60}")
    print("【角色卡结果】")

    direct_answer = final_state.get("direct_answer")
    character_cards = []

    if direct_answer:
        print(direct_answer)
    else:
        character_cards = final_state.get("character_cards") or {}
        if isinstance(character_cards, dict):
            character_cards = list(character_cards.values())
        if character_cards:
            print(f"\n共生成 {len(character_cards)} 张角色卡：\n")
            for i, card in enumerate(character_cards, 1):
                name = card.get("name", "?")
                period = card.get("period", "全程")
                print(f"--- #{i} {name} ({period}) ---")
                print(json.dumps(card, ensure_ascii=False, indent=2))
                print()
        else:
            print("（未能生成任何角色卡）")

    if final_state.get("quality_score") is not None:
        print(f"\n质量评分: {final_state['quality_score']:.2f}")
    if final_state.get("total_tokens_used"):
        print(f"总 token 消耗: {final_state['total_tokens_used']}")
    steps = final_state.get("execution_steps", [])
    if steps:
        print("\n执行步骤:")
        for s in steps:
            print(f"  {s}")

    if args.output_dir and character_cards:
        char_output = os.path.join(args.output_dir, "characters.json")
        _safe_write_json(char_output, character_cards)
        print(f"\n角色卡已保存至: {char_output}")

    narrator_card = final_state.get("narrator_card")
    if narrator_card:
        print(f"\n--- 旁白角色卡 ---")
        print(json.dumps(narrator_card, ensure_ascii=False, indent=2))
        print()
        if args.output_dir:
            narrator_output = os.path.join(args.narrator_output_dir or args.output_dir, "narrator.json")
            _safe_write_json(narrator_output, narrator_card)
            print(f"旁白卡已保存至: {narrator_output}")

    print(f"{'='*60}")

    if args.tts and character_cards:
        _generate_tts(character_cards, args.tts_dir, args.tts_text)


def cmd_tts(args):
    """从角色卡 JSON 文件生成样音"""
    from src.utils.tts_engine import generate_sample_audio_from_json

    if not os.path.isfile(args.cards):
        print(f"角色卡文件不存在: {args.cards}")
        return

    print(f"\n{'='*60}")
    print(f"角色卡文件: {args.cards}")
    print(f"输出目录: {args.tts_dir}")
    print(f"{'='*60}\n")

    files = generate_sample_audio_from_json(args.cards, output_dir=args.tts_dir, sample_text=args.tts_text)

    print(f"\n生成 {len(files)} 个音频文件:")
    for f in files:
        print(f"  {f}")
    print(f"{'='*60}")


def _generate_tts(character_cards, tts_dir, tts_text):
    """在生成角色卡后直接生成样音"""
    from src.utils.tts_engine import generate_sample_audio

    print(f"\n{'='*60}")
    print("【生成角色样音】")
    files = generate_sample_audio(character_cards, output_dir=tts_dir, sample_text=tts_text)
    print(f"\n生成 {len(files)} 个音频文件:")
    for f in files:
        print(f"  {f}")
    print(f"{'='*60}")


def cmd_narrator(args):
    """生成旁白角色卡"""
    from src.utils.tts_engine import generate_narrator_card
    from src.utils.doc_parser import parse_file

    if not os.path.isfile(args.cards):
        print(f"角色卡文件不存在: {args.cards}")
        return

    with open(args.cards, "r", encoding="utf-8") as f:
        cards = json.load(f)
    if isinstance(cards, dict):
        cards = list(cards.values())

    sample_text = ""
    if args.file and os.path.isfile(args.file):
        sample_text = parse_file(args.file)[:2000]
    else:
        sample_text = "（无小说样本，请根据角色卡推断旁白风格）"

    print(f"\n{'='*60}")
    print("【生成旁白角色卡】")
    narrator, tokens = generate_narrator_card(cards, sample_text)
    if narrator:
        print(json.dumps(narrator, ensure_ascii=False, indent=2))
        if args.output_dir:
            narrator_output = os.path.join(args.output_dir, "narrator.json")
            _safe_write_json(narrator_output, narrator)
            print(f"\n旁白卡已保存至: {args.output_dir}")
    else:
        print("旁白角色卡生成失败")
    print(f"{'='*60}")


def cmd_narrate(args):
    """分拣对话并合成多角色朗读音频"""
    from src.utils.tts_engine import segment_dialogue, generate_narrated_audio, generate_narrator_card
    from src.utils.doc_parser import parse_file

    if not os.path.isfile(args.cards):
        print(f"角色卡文件不存在: {args.cards}")
        return

    with open(args.cards, "r", encoding="utf-8") as f:
        cards = json.load(f)
    if isinstance(cards, dict):
        cards = list(cards.values())

    novel_text = parse_file(args.file)
    if args.max_chars > 0:
        novel_text = novel_text[10000:10000 + args.max_chars]

    print(f"\n{'='*60}")
    print(f"小说文件: {args.file}")
    print(f"文本长度: {len(novel_text)} 字符")
    print(f"{'='*60}\n")

    # 1. 旁白角色卡：优先加载已有文件，否则现场生成
    narrator = None
    if args.narrator and os.path.isfile(args.narrator):
        print("【1/3 加载已有旁白角色卡】")
        with open(args.narrator, "r", encoding="utf-8") as f:
            narrator = json.load(f)
        if narrator:
            print(f"  旁白风格: {narrator.get('narrator_style', '?')}")
        else:
            print("  旁白卡文件为空，将现场生成")
    else:
        print("【1/3 生成旁白角色卡】")
        narrator, tokens = generate_narrator_card(cards, novel_text[:2000])
        if narrator:
            print(f"  旁白风格: {narrator.get('narrator_style', '?')}")
        else:
            print("  旁白卡生成失败，使用默认旁白声线")

    # 2. 分拣对话
    print("\n【2/3 分拣旁白与角色对话】")
    segments = segment_dialogue(novel_text, cards)
    narrator_count = sum(1 for s in segments if s["speaker"] == "旁白")
    print(f"  共 {len(segments)} 段：旁白 {narrator_count} 段，角色对话 {len(segments) - narrator_count} 段")
    for i, s in enumerate(segments[:5]):
        print(f"  [{i+1}] {s['speaker']}: {s['text'][:40]}...")
    if len(segments) > 5:
        print(f"  ... (共 {len(segments)} 段)")

    # 3. 合成音频
    print("\n【3/3 合成多角色朗读音频】")
    output = generate_narrated_audio(segments, cards, narrator, output_dir=args.tts_dir, output_filename=args.output_file)
    if output:
        print(f"\n朗读音频: {output}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="小说角色卡生成 Agent")
    subparsers = parser.add_subparsers(dest="command")

    # 默认命令：生成角色卡
    gen_parser = subparsers.add_parser("generate", help="解析小说并生成角色卡")
    gen_parser.add_argument("--file", "-f", type=str, default="", help="小说文件路径")
    gen_parser.add_argument("--requirement", "-r", type=str, default="", help="角色卡生成需求")
    gen_parser.add_argument("--max-retries", type=int, default=3, help="最大重试次数")
    gen_parser.add_argument("--output-dir", "-o", type=str, default="./output/char", help="角色卡 JSON 输出路径")
    gen_parser.add_argument("--tts", action="store_true", help="生成角色卡后直接生成样音")
    gen_parser.add_argument("--tts-dir", type=str, default="./output/audio", help="样音输出目录")
    gen_parser.add_argument("--tts-text", type=str, default="", help="样音文本（默认使用通用测试句）")
    gen_parser.add_argument("--narrator-output-dir", type=str, default="./output/char", help="旁白卡 JSON 输出路径（默认与角色卡同目录）")

    # TTS 命令：从已有角色卡 JSON 生成样音
    tts_parser = subparsers.add_parser("tts", help="从角色卡 JSON 生成样音")
    tts_parser.add_argument("--cards", "-c", type=str, required=True, help="角色卡 JSON 文件路径")
    tts_parser.add_argument("--tts-dir", "-d", type=str, default="./output/audio", help="样音输出目录")
    tts_parser.add_argument("--tts-text", "-t", type=str, default="", help="样音文本")

    # 旁白卡生成命令
    nar_parser = subparsers.add_parser("narrator", help="生成旁白角色卡")
    nar_parser.add_argument("--cards", "-c", type=str, required=True, help="角色卡 JSON 文件路径")
    nar_parser.add_argument("--file", "-f", type=str, default="", help="小说文件路径（提供叙事风格参考）")
    nar_parser.add_argument("--output-dir", "-o", type=str, default="./output/char", help="旁白卡 JSON 输出路径")

    # 多角色朗读命令
    narrate_parser = subparsers.add_parser("narrate", help="分拣对话并合成多角色朗读音频")
    narrate_parser.add_argument("--file", "-f", type=str, default="../家丁.txt", help="小说文件路径")
    narrate_parser.add_argument("--cards", "-c", type=str, default="./output/char/characters.json", help="角色卡 JSON 文件路径")
    narrate_parser.add_argument("--tts-dir", "-d", type=str, default="./output/audio", help="音频输出目录")
    narrate_parser.add_argument("--output-file", type=str, default="narrated.mp3", help="输出音频文件名")
    narrate_parser.add_argument("--max-chars", type=int, default=1500, help="最多处理的字符数（0=全部）")
    narrate_parser.add_argument("--narrator", "-n", type=str, default="./output/char/narrator.json", help="已有旁白卡 JSON 文件路径（不提供则自动生成）")

    args = parser.parse_args()

    if args.command == "tts":
        cmd_tts(args)
    elif args.command == "narrator":
        cmd_narrator(args)
    elif args.command == "narrate":
        cmd_narrate(args)
    elif args.command == "generate":
        cmd_generate(args)
    else:
        # args = gen_parser.parse_args()
        args = narrate_parser.parse_args()
        # cmd_generate(args)
        cmd_narrate(args)


if __name__ == "__main__":
    main()
