"""
tools/definitions.py - 툴 정의 (스키마 + 구현 + 레지스트리)

LLM에게 노출할 툴들을 정의하는 파일. 각 툴은 두 부분으로 구성된다.

1. 스키마 (TOOL_SCHEMAS)
   - OpenAI function calling 포맷의 JSON 스키마.
   - LLM은 이 스키마의 name/description/parameters만 보고 툴을 선택한다.
   - 따라서 description에 도메인 용어와 "반환값에 어떤 ID가 포함되는지"를
     명확히 적어야 체이닝 정확도가 올라간다. (설계의 핵심!)

2. 구현 (tool_* 함수들)
   - 실제 데이터 조회 로직. mes_mock의 접근 함수를 호출하고,
     결과를 "LLM이 읽기 좋은 컴팩트한 dict"로 정규화해서 반환한다.
   - 반환값은 반드시 JSON 직렬화 가능해야 한다. (graph.py에서 json.dumps 함)
   - 에러(없는 ID 등)도 예외 대신 {"error": ...} dict로 반환한다.
     -> LLM이 에러 내용을 보고 스스로 정정(오타 수정, 재검색)할 수 있게 하기 위함.

[새 툴 추가 방법]
  1) tool_xxx 함수 작성
  2) TOOL_SCHEMAS에 스키마 추가
  3) TOOL_REGISTRY에 {"이름": 함수} 등록
  -> graph.py는 수정할 필요 없음 (레지스트리 기반으로 동작)
"""

# 파이썬 3.9 호환: "str | None" 표기(PEP 604)를 쓰기 위한 지연 평가
from __future__ import annotations

from typing import Any, Callable

from tools import mes_mock

# =============================================================================
# 1. 툴 구현
# =============================================================================

def tool_get_equipment_status(eqp_id: str) -> dict[str, Any]:
    """장비 1대의 현재 상태를 조회한다."""
    eqp = mes_mock.fetch_equipment(eqp_id)
    if eqp is None:
        # 없는 장비 -> 에러와 함께 전체 장비 ID 목록을 힌트로 준다.
        # LLM이 사용자의 오타를 스스로 교정해서 재시도할 수 있다.
        return {
            "error": f"장비 '{eqp_id}'를 찾을 수 없습니다.",
            "hint_all_eqp_ids": [e["eqp_id"] for e in mes_mock._EQUIPMENTS.values()],
        }
    # 목업 데이터를 그대로 반환해도 되지만, 실서버라면 여기서
    # 불필요한 필드를 제거하고 필요한 것만 추리는 정규화를 수행한다.
    return eqp


def tool_get_lot_info(lot_id: str) -> dict[str, Any]:
    """랏 1개의 현재 위치/상태(WIP)를 조회한다."""
    lot = mes_mock.fetch_lot(lot_id)
    if lot is None:
        return {"error": f"랏 '{lot_id}'를 찾을 수 없습니다."}
    return lot


def tool_get_lot_metrology(lot_id: str, step: str | None = None) -> dict[str, Any]:
    """랏의 계측(측정) 결과 이력을 조회한다."""
    # 랏 존재 여부를 먼저 확인해서, "계측 이력이 없는 것"과
    # "랏 자체가 없는 것"을 구분해 준다. (LLM의 오답 방지)
    if mes_mock.fetch_lot(lot_id) is None:
        return {"error": f"랏 '{lot_id}'를 찾을 수 없습니다."}

    records = mes_mock.fetch_metrology(lot_id, step)
    return {
        "lot_id": lot_id,
        "filter_step": step,          # 어떤 필터로 조회했는지 echo (LLM 혼동 방지)
        "count": len(records),
        "records": records,
    }


def tool_find_capable_equipment(process_step: str) -> dict[str, Any]:
    """특정 공정 스텝을 진행할 수 있는 장비 목록을 조회한다."""
    equipments = mes_mock.fetch_equipments_by_step(process_step)
    if not equipments:
        return {
            "error": f"공정 스텝 '{process_step}'을 진행 가능한 장비가 없습니다.",
            "hint": "스텝 이름 예시: PHOTO, ETCH, METRO_CD, METRO_THK",
        }
    return {
        "process_step": process_step,
        "count": len(equipments),
        "equipments": [
            {
                "eqp_id": e["eqp_id"],
                "name": e["name"],
                "status": e["status"],
                # "지금 당장 배정 가능한가"를 미리 계산해 준다.
                # LLM이 상태 코드의 의미를 해석하다 틀리는 것을 방지.
                "available_now": e["status"] == "IDLE",
            }
            for e in equipments
        ],
    }


def tool_search_lots(
    eqp_id: str | None = None,
    step: str | None = None,
    state: str | None = None,
) -> dict[str, Any]:
    """조건(장비/스텝/상태)에 맞는 랏들을 검색한다."""
    lots = mes_mock.fetch_lots_filtered(eqp_id=eqp_id, step=step, state=state)
    return {
        # 어떤 조건으로 검색했는지 echo. 병렬 호출 시 결과 매칭에 도움이 된다.
        "filters": {"eqp_id": eqp_id, "step": step, "state": state},
        "count": len(lots),
        "lots": lots,
    }


# =============================================================================
# 2. 툴 스키마 (LLM에게 보여지는 부분)
#
# description 작성 원칙:
#  - 언제 이 툴을 써야 하는지 (트리거 상황)
#  - 반환값에 어떤 ID/필드가 들어있는지 -> 체이닝 유도
#  - 도메인 용어의 의미
# =============================================================================

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_equipment_status",
            "description": (
                "반도체 장비 1대의 현재 상태를 조회한다. "
                "상태(RUN=가동중/IDLE=대기/DOWN=고장/PM=예방정비), "
                "현재 진행 중인 랏 ID(current_lot_id), "
                "최근 완료한 랏 ID 목록(recent_lot_ids, 최신순), "
                "진행 가능한 공정 스텝(capable_steps)을 반환한다. "
                "반환된 랏 ID는 get_lot_info나 get_lot_metrology에 "
                "그대로 넣어 후속 조회할 수 있다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "eqp_id": {
                        "type": "string",
                        "description": "장비 ID (예: EQP-PHO-01)",
                    },
                },
                "required": ["eqp_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lot_info",
            "description": (
                "랏(웨이퍼 묶음) 1개의 현재 상태를 조회한다. "
                "현재 위치한 공정 스텝(current_step), "
                "현재 있는 장비 ID(current_eqp_id, 대기 중이면 null), "
                "진행 상태(WAIT=스텝 대기/PROC=진행중/HOLD=홀드), "
                "홀드 사유(hold_reason)를 반환한다. "
                "반환된 장비 ID는 get_equipment_status로 후속 조회할 수 있다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lot_id": {
                        "type": "string",
                        "description": "랏 ID (예: LOT-A001)",
                    },
                },
                "required": ["lot_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lot_metrology",
            "description": (
                "랏의 계측(metrology, 측정) 결과 이력을 조회한다. "
                "각 레코드는 계측 스텝, 결과(PASS=스펙인/FAIL=스펙아웃/PENDING=측정대기), "
                "측정값과 스펙 범위, 측정한 장비 ID를 포함한다. "
                "랏이 홀드된 원인이 계측 스펙 아웃인지 확인할 때 유용하다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lot_id": {
                        "type": "string",
                        "description": "랏 ID (예: LOT-B001)",
                    },
                    "step": {
                        "type": "string",
                        "description": (
                            "특정 계측 스텝만 필터링 (예: METRO_CD, METRO_THK). "
                            "생략하면 전체 이력을 반환한다."
                        ),
                    },
                },
                "required": ["lot_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_capable_equipment",
            "description": (
                "특정 공정 스텝을 진행할 수 있는 장비 목록을 조회한다. "
                "각 장비의 상태와 함께, 지금 즉시 배정 가능한지 여부"
                "(available_now: IDLE 상태인 경우 true)를 반환한다. "
                "'이 공정 어디서 진행할 수 있어?' 류의 질문에 사용한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "process_step": {
                        "type": "string",
                        "description": "공정 스텝 이름 (예: PHOTO, ETCH, METRO_CD, METRO_THK)",
                    },
                },
                "required": ["process_step"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_lots",
            "description": (
                "조건에 맞는 랏들을 검색한다. 모든 조건은 선택 사항이며 "
                "AND로 결합된다. '홀드된 랏 전부 보여줘'(state=HOLD), "
                "'EQP-MET-01에서 진행 중인 랏은?'(eqp_id 지정) 같은 "
                "목록성 질문에 사용한다. 개별 랏 ID를 이미 알고 있다면 "
                "이 툴 대신 get_lot_info를 사용할 것."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "eqp_id": {
                        "type": "string",
                        "description": "현재 이 장비에 있는 랏만 검색",
                    },
                    "step": {
                        "type": "string",
                        "description": "현재 이 공정 스텝에 있는 랏만 검색",
                    },
                    "state": {
                        "type": "string",
                        "enum": ["WAIT", "PROC", "HOLD"],
                        "description": "진행 상태로 필터링",
                    },
                },
                "required": [],
            },
        },
    },
]

# =============================================================================
# 3. 레지스트리: 툴 이름 -> 구현 함수 매핑
#
# graph.py의 툴 실행 노드는 이 dict만 보고 동작한다.
# LLM이 반환한 tool_call의 name으로 여기서 함수를 찾아 실행한다.
# =============================================================================

TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "get_equipment_status": tool_get_equipment_status,
    "get_lot_info": tool_get_lot_info,
    "get_lot_metrology": tool_get_lot_metrology,
    "find_capable_equipment": tool_find_capable_equipment,
    "search_lots": tool_search_lots,
}
