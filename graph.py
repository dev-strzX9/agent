"""
graph.py - 노드형 ReAct 그래프 조립 (추출 파이프라인 + ReAct 루프)

[구조]

        START
          │
          ▼
       extract                     1차 추출 (LLM 1회): 질문 -> queries 체크리스트
          │                        Pydantic 검증 + 정규화(대문자화, limit 클램프)
          ▼
       repair                      2차 보정 + 검증:
          │                        - 식별자 빠진 query 있을 때만 보정 LLM 호출
          │                        - 추출된 ID의 DB 실존 확인 (환각/오타 필터)
          ▼
    ┌── router ────────────────┐   판단 노드 (LLM이 queries vs results 대조)
    │     │                    │
    │     ├─ "fetch_equipment" │
    │     │      ▼             │
    │     │  fetch_equipment ──┤   실행 노드 (장비 조회, LLM 없음)
    │     │                    │
    │     ├─ "fetch_lot"       │
    │     │      ▼             │
    │     │  fetch_lot ────────┘   실행 노드 (랏 조회/팬아웃, LLM 없음)
    │     │                        실행이 끝나면 router로 "복귀" <- ReAct 루프
    │     │
    │     └─ "summarize"
    │            ▼
    │        summarize              종료 노드 (LLM이 결과를 답변으로 정리)
    │            │
    │            ▼
    └────────── END

[역할 분담]
- extract/repair: "무엇을 알고 싶은가"를 확정 (상위 계획, 안정성 장치)
- router: "체크리스트 중 다음 한 걸음"만 판단 (질문 전체 재해석 불필요)
- 실행 노드: 실제 조회. 체이닝의 데이터 연결은 router가 results에서
  ID를 읽어 params에 넣는 방식으로 일어난다.

[이게 왜 ReAct인가]
tool calling은 없지만, "판단(router) -> 실행(fetch_*) -> 관찰(results에
쌓인 걸 다음 턴에 봄) -> 다시 판단"의 왕복 루프가 있으므로 ReAct다.
extract/repair는 루프 밖의 전처리 파이프라인(직선)이다.
"""

from langgraph.graph import END, START, StateGraph

from nodes.extract import extract_node
from nodes.fetch_equipment import fetch_equipment_node
from nodes.fetch_lot import fetch_lot_node
from nodes.repair import repair_node
from nodes.router import route_next, router_node
from nodes.summarize import summarize_node
from state import AgentState


def build_graph():
    """노드형 ReAct 그래프를 조립하고 컴파일해서 반환한다.

    실행:
        app = build_graph()
        result = app.invoke({"question": "...", "results": [], "router_turns": 0})
        print(result["answer"])
    """
    builder = StateGraph(AgentState)

    # --- 노드 등록 (노드 1개 = nodes/ 아래 폴더 1개) ---
    builder.add_node("extract", extract_node)
    builder.add_node("repair", repair_node)
    builder.add_node("router", router_node)
    builder.add_node("fetch_equipment", fetch_equipment_node)
    builder.add_node("fetch_lot", fetch_lot_node)
    builder.add_node("summarize", summarize_node)

    # --- 흐름 연결 ---
    # 전처리 파이프라인 (직선): 추출 -> 보정/검증 -> 판단 루프 진입.
    # 추출이 실패해도 repair가 phase만 기록하고 router로 넘어간다.
    # (router는 체크리스트가 비어 있으면 질문 원문으로 동작 - 폴백 모드)
    builder.add_edge(START, "extract")
    builder.add_edge("extract", "repair")
    builder.add_edge("repair", "router")

    # router의 결정(next_node)에 따라 분기.
    # 새 실행 노드를 추가하면 여기 매핑에 1줄 추가하면 된다.
    builder.add_conditional_edges(
        "router",
        route_next,
        {
            "fetch_equipment": "fetch_equipment",
            "fetch_lot": "fetch_lot",
            "summarize": "summarize",
        },
    )

    # 실행 노드가 끝나면 router로 복귀 -> 이 엣지가 ReAct 루프를 만든다.
    builder.add_edge("fetch_equipment", "router")
    builder.add_edge("fetch_lot", "router")

    # summarize가 답변을 만들면 종료.
    builder.add_edge("summarize", END)

    return builder.compile()
