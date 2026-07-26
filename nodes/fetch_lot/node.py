"""
nodes/fetch_lot - 실행 노드: 랏 조회 (ReAct의 Act 단계, 팬아웃 지원)

router가 {"next": "fetch_lot", "params": {"lot_ids": ["LOT-A001", ...]}}를
결정하면 이 노드가 실행된다. LLM은 전혀 쓰지 않는다.

팬아웃:
  lot_ids가 여러 개면 스레드풀로 "동시에" 조회한다.
  ("장비 A가 최근 진행한 랏들 위치 전부"처럼 1->N으로 퍼지는 조회)
  랏이 몇 개든 반복문이 처리하므로, tool calling ReAct에서 약한 모델이
  일부 랏만 조회하고 넘어가는 류의 실수가 원천적으로 없다.

체이닝과의 관계:
  이 노드는 lot_ids가 어디서 왔는지 모른다. 사용자 질문에 있던 ID일 수도,
  직전 fetch_equipment 결과의 recent_lot_ids일 수도 있다.
  그 연결(결과에서 ID를 꺼내 params에 넣기)은 router(LLM)의 몫이다.
"""

# 파이썬 3.9 호환
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from config import MAX_FANOUT_WORKERS
from mes import mock
from state import AgentState


def _fetch_one(lot_id: str) -> dict[str, Any]:
    """랏 1개 조회. 없으면 에러 dict (예외를 던지지 않는다)."""
    lot = mock.fetch_lot(lot_id)
    if lot is None:
        return {"error": f"랏 '{lot_id}'를 찾을 수 없습니다."}
    return lot


def fetch_lot_node(state: AgentState) -> dict[str, Any]:
    """랏 N개를 병렬 조회하고 결과를 results에 쌓는다."""
    params = state.get("next_params", {})

    # router가 단수형("lot_id")으로 줄 수도 있으므로 둘 다 받아준다.
    lot_ids = params.get("lot_ids") or []
    if not lot_ids and params.get("lot_id"):
        lot_ids = [params["lot_id"]]

    if not lot_ids:
        result: dict[str, Any] = {
            "error": "lot_ids 파라미터가 없습니다.",
            "hint_all_lot_ids": mock.all_lot_ids(),
        }
        lots: list[dict[str, Any]] = []
    else:
        # 팬아웃: 각 랏 조회는 서로 독립이므로 동시에 실행한다.
        # (목업은 즉시 반환이지만, 실서버 MES는 건당 수백 ms라 효과가 크다)
        with ThreadPoolExecutor(max_workers=min(len(lot_ids), MAX_FANOUT_WORKERS)) as pool:
            # pool.map은 입력 순서를 보존한다 -> lot_ids[i] <-> lots[i] 대응
            lots = list(pool.map(_fetch_one, lot_ids))
        result = {"count": len(lots), "lots": lots}

    return {
        "results": [
            {"node": "fetch_lot", "params": {"lot_ids": lot_ids}, "result": result}
        ]
    }
