import argparse

from src.core.graph import build_agent_graph
from src.core.state import AgentState


def main():
    parser = argparse.ArgumentParser(description="多Agent长文本解析系统")
    parser.add_argument("--input", "-i", type=str, default="", help="用户问题/需求")
    parser.add_argument("--context", "-c", type=str, default="", help="上下文/文档内容")
    parser.add_argument("--context-file", "-f", type=str, default="", help="上下文文件路径（大文件推荐）")
    parser.add_argument("--max-retries", type=int, default=3, help="最大重试次数")

    args = parser.parse_args()

    if not args.input:
        args.input = input("请输入你的问题/需求: ")

    context = args.context
    context_path = args.context_file

    if not context and not context_path:
        try:
            context = input("请输入上下文/文档内容(直接回车跳过): ")
        except EOFError:
            context = ""

    print(f"\n{'='*60}")
    print(f"问题: {args.input}")
    if context:
        print(f"上下文长度: {len(context)} 字符")
    if context_path:
        print(f"上下文文件: {context_path}")
    print(f"{'='*60}\n")

    graph = build_agent_graph()

    initial_state = AgentState(
        original_input=args.input,
        raw_context=context,
        context_path=context_path,
        max_retries=args.max_retries,
    )

    final_state = graph.invoke(initial_state)

    print(f"\n{'='*60}")
    print("【最终结果】")

    if final_state.get("direct_answer"):
        print(final_state["direct_answer"])
    elif final_state.get("final_answer"):
        print(final_state["final_answer"])
    elif final_state.get("worker_results"):
        results = final_state["worker_results"]
        if isinstance(results, dict):
            for agent_id, content in results.items():
                print(f"\n--- {agent_id} ---")
                print(content)
        else:
            print(results)
    else:
        print("（无输出结果）")

    if final_state.get("quality_score"):
        print(f"\n质量评分: {final_state['quality_score']:.2f}")
    if final_state.get("total_tokens_used"):
        print(f"总 token 消耗: {final_state['total_tokens_used']}")
    if final_state.get("total_cost"):
        print(f"总费用: ${final_state['total_cost']:.6f}")
    steps = final_state.get("execution_steps", [])
    if steps:
        print("\n执行步骤:")
        for s in steps:
            print(f"  {s}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()