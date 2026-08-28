"""
nodes/fetch_equipment - 실행 노드: 장비 조회 (ReAct의 Act 단계)

router가 {"next": "fetch_equipment", "params": {"eqp_id": "..."}}를
결정하면 이 노드가 실행된다. LLM은 전혀 쓰지 않는다.

역할:
  1. state["next_params"]에서 eqp_id를 꺼낸다
  2. MES에서 장비 정보를 조회한다
  3. 결과를 state["results"]에 append 한다 (router가 다음 턴에 보게 됨)

조회 결과에 recent_lot_ids / current_lot_id가 포함되므로,
router(LLM)가 다음 턴에 이 ID들로 fetch_lot을 선택하면 체이닝이 일어난다.

에러 처리:
  존재하지 않는 장비여도 예외를 던지지 않고 {"error": ..., "hint": ...}를
  results에 넣는다. router가 힌트(전체 장비 목록)를 보고 오타를 정정해
  재시도하거나, summarize가 "장비 없음"을 답변으로 설명할 수 있다.
"""

# 파이썬 3.9 호환
from __future__ import annotations

from typing import Any

from mes import mock
from state import AgentState


def fetch_equipment_node(state: AgentState) -> dict[str, Any]:
    """장비 1대를 조회하고 결과를 results에 쌓는다."""
    params = state.get("next_params", {})
    eqp_id = params.get("eqp_id")

    # router(LLM)가 params를 빼먹었을 때의 방어. 에러도 결과로 남겨서
    # router가 다음 턴에 정정할 수 있게 한다.
    if not eqp_id:
        result: dict[str, Any] = {
            "error": "eqp_id 파라미터가 없습니다.",
            "hint_all_eqp_ids": mock.all_equipment_ids(),
        }
    else:
        eqp = mock.fetch_equipment(eqp_id)
        if eqp is None:
            result = {
                "error": f"장비 '{eqp_id}'를 찾을 수 없습니다.",
                "hint_all_eqp_ids": mock.all_equipment_ids(),
            }
        else:
            result = eqp

    # results에 "무엇을 조회했고 결과가 뭔지"를 한 건으로 기록한다.
    # (node/params를 함께 남겨야 router가 "이 조회는 이미 했다"를 안다)
    return {
        "results": [
            {"node": "fetch_equipment", "params": {"eqp_id": eqp_id}, "result": result}
        ]
    }
