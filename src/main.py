import argparse
import json

from src.core.graph import build_agent_graph
from src.core.state import AgentState


def main():
    parser = argparse.ArgumentParser(description="小说角色卡生成 Agent")
    parser.add_argument("--file", "-f", type=str, default="", help="小说 TXT 文件路径")
    parser.add_argument("--requirement", "-r", type=str, default="", help="角色卡生成需求（可留空，默认生成全部角色）")
    parser.add_argument("--max-retries", type=int, default=3, help="质量不合格时最大重试次数")

    args = parser.parse_args()

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
    print(f"{'='*60}")


if __name__ == "__main__":
    main()