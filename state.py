"""
state.py - 그래프 공유 상태 정의

노드형 ReAct의 핵심은 state다. 판단 노드(router)는 매번 이 state를 보고
"질문 대비 아직 부족한 조회가 뭔지"를 판단한다.

[tool calling ReAct와의 차이]
- tool calling: 판단 근거가 messages(대화 이력 + role:"tool" 결과)
- 노드형:       판단 근거가 구조화된 state (question + results)
  LLM에게 매번 state를 JSON으로 보여주고 "다음 노드"만 고르게 한다.

[results에 쌓이는 것]
실행 노드(fetch_equipment, fetch_lot)가 끝날 때마다 1건씩 append 된다.
    {"node": "fetch_equipment", "params": {"eqp_id": "..."}, "result": {...}}
router는 이 목록을 보고 "장비는 조회했고 랏은 아직이네"를 판단한다.
"""

# 파이썬 3.9 호환
from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """에이전트 그래프의 공유 상태.

    Attributes:
        question: 사용자의 원래 질문. 추출/판단/요약 모두의 기준.
        queries: extract/repair가 만든 "질의 체크리스트". (덮어쓰기)
            각 원소: {"eqp_id": ..., "needs": [...], "lot_limit": ...}
            DB 검증 실패 시 validation_error/hint가 붙어 있다.
            router는 이 목록과 results를 대조해 "뭐가 남았는지" 판단한다.
        phase: 추출 파이프라인의 진행 단계. (덮어쓰기, 디버깅/로그용)
            "extracted" -> "validated" 또는 "extraction_failed"
        phase_message: phase에 대한 사람이 읽을 설명. (덮어쓰기)
        results: 지금까지 실행한 조회 결과 목록. (append-only, 리듀서로 누적)
            각 원소: {"node": 노드이름, "params": 입력, "result": 조회결과}
        next_node: router가 결정한 다음 노드 이름. (덮어쓰기)
            "fetch_equipment" | "fetch_lot" | "summarize"
        next_params: 다음 노드에 넘길 파라미터. (덮어쓰기)
            예: {"eqp_id": "EQP-PHO-01"} 또는 {"lot_ids": ["LOT-A001"]}
        router_turns: router가 실행된 횟수. 무한 루프 방지용. (누적)
        answer: summarize가 작성한 최종 답변.
    """

    question: str
    queries: list[dict[str, Any]]
    phase: str
    phase_message: str
    results: Annotated[list[dict[str, Any]], operator.add]
    next_node: str
    next_params: dict[str, Any]
    router_turns: Annotated[int, operator.add]
    answer: str
