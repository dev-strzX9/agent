"""
nodes/repair - 2차 보정 + 검증 노드 (stage-2)

ermap_agent의 repair_task_identifiers_node에 해당. 1차 추출(extract)의
queries를 받아 3중 방어를 완성한다.

  1) 조건부 식별자 보정 (LLM, 필요할 때만)
     식별자(eqp_id/lot_id)가 빠진 query가 있을 때"만" 보정 LLM을 호출한다.
     1차 추출이 needs는 잘 뽑았는데 ID를 빠뜨리는 경우가 있어서,
     질문 원문 + 1차 결과를 함께 주고 식별자만 다시 찾게 한다.
     -> 평소에는 LLM 0회, 문제가 있을 때만 1회 (비용 합리적)

  2) 실존 확인 (DB)
     추출/보정된 ID가 실제 DB에 있는지 확인한다. LLM 환각/사용자 오타가
     여기서 걸러진다. 없는 ID면 query에 validation_error + 전체 목록
     힌트를 붙인다. (query를 버리지 않는다 -> summarize가 "해당 장비
     없음, 혹시 이 중 하나?"로 답변할 수 있게)

  3) phase 기록
     validated / extraction_failed 를 state에 남겨 어디까지 성공했는지
     로그에서 바로 보이게 한다.

정규화(대문자화 등)는 이미 1단계: Pydantic field_validator가 수행했다.
(schemas.py 참고 - 표기 정규화(코드) -> 식별자 보정(LLM) -> 실존 확인(DB))
"""

# 파이썬 3.9 호환
from __future__ import annotations

import json
from typing import Any

from llm_client import chat_structured
from mes import mock
from schemas import RepairQueries
from state import AgentState

_REPAIR_SYSTEM_PROMPT = """\
당신은 반도체 MES 질의의 식별자 보정기입니다.
1차 추출된 queries 중 일부에 식별자(eqp_id 또는 lot_id)가 빠져 있습니다.
사용자 질문 원문을 다시 읽고, 각 query에 해당하는 식별자를 찾아 채우세요.

[규칙]
1. 입력 queries와 "같은 개수, 같은 순서"로 출력한다.
2. 각 항목에는 식별자 필드(eqp_id, lot_id)만 넣는다.
3. 이미 식별자가 있는 query는 그대로 두면 되므로 빈 객체 {}를 출력한다.
4. 질문에 없는 식별자를 지어내지 않는다. 정말 없으면 빈 객체 {}를 출력한다.
"""


def _needs_repair(queries: list[dict[str, Any]]) -> bool:
    """식별자가 없는 query가 하나라도 있는가."""
    return any(not (q.get("eqp_id") or q.get("lot_id")) for q in queries)


def _merge_repair(
    queries: list[dict[str, Any]],
    repairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """보정 결과를 1차 queries에 병합한다.

    "비어 있는 필드만" 채운다. 1차 추출이 이미 찾은 식별자를
    보정 LLM이 덮어쓰지 못하게 하기 위함이다. (1차 결과 우선)
    """
    merged = []
    for i, q in enumerate(queries):
        item = dict(q)
        if i < len(repairs):
            for key in ("eqp_id", "lot_id"):
                if not item.get(key) and repairs[i].get(key):
                    item[key] = repairs[i][key]
        merged.append(item)
    return merged


def _validate_against_db(query: dict[str, Any]) -> dict[str, Any]:
    """식별자가 DB에 실존하는지 확인하고, 없으면 에러+힌트를 붙인다.

    에러가 있어도 query를 버리지 않는다. router가 이 query를 건너뛰고,
    summarize가 힌트(전체 목록)와 함께 "찾을 수 없음"을 답변하게 된다.
    """
    if query.get("eqp_id") and mock.fetch_equipment(query["eqp_id"]) is None:
        query["validation_error"] = f"장비 '{query['eqp_id']}'가 DB에 존재하지 않습니다."
        query["hint"] = {"all_eqp_ids": mock.all_equipment_ids()}
    elif query.get("lot_id") and mock.fetch_lot(query["lot_id"]) is None:
        query["validation_error"] = f"랏 '{query['lot_id']}'가 DB에 존재하지 않습니다."
        query["hint"] = {"all_lot_ids": mock.all_lot_ids()}
    return query


def repair_node(state: AgentState) -> dict[str, Any]:
    """queries를 보정/검증하고 phase를 기록한다."""
    queries = [dict(q) for q in state.get("queries", [])]

    # extract가 실패했거나 질의가 없으면 보정할 것도 없다.
    # (router가 질문 원문으로 동작하는 폴백 모드)
    if not queries:
        return {
            "phase": state.get("phase", "extraction_failed"),
            "phase_message": "보정할 질의 없음 (router가 질문 원문으로 동작)",
        }

    # --- 1) 조건부 식별자 보정 (필요할 때만 LLM 1회) --------------------------
    if _needs_repair(queries):
        parsed = chat_structured(
            system_prompt=_REPAIR_SYSTEM_PROMPT,
            user_content=json.dumps(
                {"user_query": state["question"], "queries": queries},
                ensure_ascii=False,
                indent=2,
            ),
            response_model=RepairQueries,
        )
        if parsed is not None:
            repairs = [r.model_dump(exclude_none=True) for r in parsed.queries]
            # 개수가 다르면 보정 결과를 신뢰하지 않는다 (순서 매칭 불가).
            if len(repairs) == len(queries):
                queries = _merge_repair(queries, repairs)

    # --- 보정 후에도 식별자가 없는 query는 제외 -------------------------------
    # (조회 자체를 시작할 수 없으므로. 몇 건이 제외됐는지는 phase_message에 남긴다)
    valid = [q for q in queries if q.get("eqp_id") or q.get("lot_id")]
    dropped = len(queries) - len(valid)

    if not valid:
        return {
            "queries": [],
            "phase": "extraction_failed",
            "phase_message": "조회에 필요한 식별자(장비/랏 ID)를 찾을 수 없습니다.",
        }

    # --- 2) DB 실존 확인 ------------------------------------------------------
    validated = [_validate_against_db(q) for q in valid]

    message = f"질의 {len(validated)}건 검증 완료"
    if dropped:
        message += f" (식별자 없는 질의 {dropped}건 제외)"
    error_count = sum(1 for q in validated if "validation_error" in q)
    if error_count:
        message += f" (DB에 없는 식별자 {error_count}건)"

    return {
        "queries": validated,
        "phase": "validated",
        "phase_message": message,
    }
