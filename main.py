"""
main.py - 실행 진입점 (대화형 CLI)

실행 방법:
    cd react_agent
    export LLM_API_KEY="..."      # 필요 시 LLM_API_BASE, LLM_MODEL도 설정
    python main.py

예시 질문:
    - EQP-PHO-01 상태 알려줘
    - EQP-MET-01이 최근에 진행한 랏들 지금 어디 있어?   <- 체이닝 (장비 -> 랏들)
    - LOT-B001 상태 알려줘

VERBOSE=True면 router가 매 턴 어떤 노드를 골랐고 조회 결과가 뭔지
콘솔에 출력된다. ReAct 루프가 도는 과정을 눈으로 따라갈 수 있다.
"""

# 파이썬 3.9 호환
from __future__ import annotations

import json
import sys

from graph import build_graph
from llm_client import LLMAPIError

VERBOSE = True


def print_step_trace(chunk: dict) -> None:
    """그래프 실행 중간 과정 출력.

    stream_mode="updates"는 노드가 하나 끝날 때마다
    {노드이름: 상태변경분} 형태의 chunk를 내보낸다.
    """
    for node_name, update in chunk.items():
        if update is None:
            continue

        if node_name in ("extract", "repair"):
            # 추출 파이프라인의 진행 단계와 체크리스트
            print(f"  [{node_name}] phase={update.get('phase')} - {update.get('phase_message', '')}")
            for q in update.get("queries", []):
                mark = " (검증 실패)" if "validation_error" in q else ""
                print(f"    query: {json.dumps(q, ensure_ascii=False)}{mark}")

        elif node_name == "router":
            # router의 결정: 다음에 어디로 갈지
            print(f"  [판단] next={update.get('next_node')} "
                  f"params={json.dumps(update.get('next_params', {}), ensure_ascii=False)}")

        elif node_name in ("fetch_equipment", "fetch_lot"):
            # 실행 노드의 조회 결과 미리보기
            for r in update.get("results", []):
                preview = json.dumps(r["result"], ensure_ascii=False)
                if len(preview) > 200:
                    preview = preview[:200] + "..."
                print(f"  [실행] {r['node']} -> {preview}")


def run_once(app, question: str) -> None:
    """질문 1건을 그래프로 처리하고 답변을 출력한다."""
    initial_state = {
        "question": question,
        "queries": [],
        "results": [],
        "router_turns": 0,
    }

    answer = ""
    if VERBOSE:
        for chunk in app.stream(initial_state, stream_mode="updates"):
            print_step_trace(chunk)
            # summarize 노드의 answer를 잡아둔다.
            for update in chunk.values():
                if update and "answer" in update:
                    answer = update["answer"]
    else:
        final_state = app.invoke(initial_state)
        answer = final_state.get("answer", "")

    print(f"\n[답변]\n{answer}\n")


def main() -> None:
    app = build_graph()

    print("=" * 60)
    print("반도체 MES 조회 에이전트 - 노드형 ReAct (종료: quit 또는 Ctrl+C)")
    print("=" * 60)
    print("예시 질문:")
    print("  - EQP-PHO-01 상태 알려줘")
    print("  - EQP-MET-01이 최근에 진행한 랏들 지금 어디 있어?")
    print()

    while True:
        try:
            question = input("질문> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n종료합니다.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("종료합니다.")
            break

        try:
            run_once(app, question)
        except LLMAPIError as e:
            print(f"\n[오류] {e}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
