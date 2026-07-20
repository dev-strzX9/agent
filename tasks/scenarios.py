"""
tasks/scenarios.py - 시나리오 함수 (하이브리드 구조의 "주 경로")

[이 파일이 존재하는 이유]
회사 LLM(GLM 낮은 버전)은 ReAct 루프에서 매 턴 요구되는 판단
(툴 선택, 인자 JSON 생성, 앞 결과에서 필드 꺼내기)을 자주 틀린다.
그래서 자주 나오는 "추적 경로"를 파이썬 함수로 굳혀서,
체이닝/팬아웃을 코드가 100% 정확하게 수행하게 한다.

LLM의 역할은 두 가지로 축소된다.
  1) 질문을 태스크 유형으로 분류 + 질문에 명시된 파라미터 추출
     (tasks/extraction.py)
  2) 시나리오 실행 결과를 자연어 답변으로 요약 (graph.py의 summarize 노드)

[태스크의 단위 = "추적 경로", 함수 호출이 아님]
"장비 최근 랏 5개의 현재 위치"라는 질문에서 랏 ID들은 질문에 없다.
장비 조회 결과 안에서 "실행 중에" 생겨나는 파생 값이기 때문이다.
그래서 태스크를 함수 호출 단위로 쪼개면 "$t1.result.recent_lot_ids"
같은 바인딩 문법이 필요해지는데(= DAG 엔진을 직접 만들게 됨),
대신 파생 값이 생기고 소비되는 구간 전체를 시나리오 하나로 묶는다.
태스크 파라미터에는 "질문에서 바로 채울 수 있는 값"만 남는다.

[체이닝이 코드로 옮겨간 모습]
    eqp = tool_get_equipment_status(eqp_id)   # 1차 조회
    lot_ids = eqp["recent_lot_ids"][:limit]   # 앞 결과에서 인자 추출 <- 코드가 함
    lots = pool.map(tool_get_lot_info, ids)   # 2차 조회 (팬아웃도 코드가 함)
LLM이 recent_lot_ids를 찾아내길 기대하는 대신 코드로 접근하므로
모델 성능과 무관하게 항상 정확하다.

[새 시나리오 추가 방법]
  1) scenario_xxx 함수 작성 (툴 함수들을 조합)
  2) SCENARIO_REGISTRY에 등록 (description은 태스크 추출 프롬프트에
     그대로 들어가므로, "어떤 질문이 이 태스크인지"를 명확히 쓸 것)
  -> extraction.py / graph.py는 수정할 필요 없음 (레지스트리 기반)

ReAct 폴백으로 자주 빠지는 질문 유형이 로그에 보이면,
그 유형을 여기에 시나리오로 "승격"시켜 주 경로를 넓혀간다.
"""

# 파이썬 3.9 호환: "str | None" 표기(PEP 604)를 쓰기 위한 지연 평가
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from tools.definitions import (
    tool_find_capable_equipment,
    tool_get_equipment_status,
    tool_get_lot_info,
    tool_get_lot_metrology,
    tool_search_lots,
)

# 팬아웃 시 동시 조회 상한 (실서버 MES 과부하 방지)
_MAX_FANOUT_WORKERS = 8


# =============================================================================
# 시나리오 함수들
#
# 공통 규칙:
# - 반환값은 JSON 직렬화 가능한 dict (요약 LLM에게 그대로 전달된다)
# - 실패해도 예외를 던지지 않고 {"error": ...}를 반환한다
#   (요약 LLM이 "해당 장비를 찾을 수 없다"고 정상적으로 답하게)
# - 질문 변형("위치만", "상태만", "홀드 여부만")을 시나리오 분화 없이
#   흡수하기 위해, 해당 엔티티에 대해 살짝 넉넉하게 조회해서 돌려준다.
#   질문에 맞는 부분만 골라 답하는 것은 요약 LLM의 몫이다.
# =============================================================================

def scenario_equipment_status(eqp_id: str) -> dict[str, Any]:
    """[단일 조회] 장비 1대의 현재 상태.

    단일 툴 호출이지만 이것도 시나리오로 등록한다.
    -> "장비 상태 알려줘" 같은 최빈 질문까지 ReAct(모델 판단)에
       맡기지 않고 주 경로에서 처리하기 위함.
    """
    return {"equipment": tool_get_equipment_status(eqp_id)}


def scenario_lot_status(lot_id: str) -> dict[str, Any]:
    """[체이닝] 랏 1개의 현재 위치/상태 + 계측 이력.

    "랏 어디 있어?" 질문에 계측까지 묶어 조회하는 이유:
    조회는 읽기 전용이라 추가 비용이 거의 없고, 후속 질문
    ("계측은 통과했어?")에 요약 LLM이 바로 답할 수 있게 된다.
    """
    lot = tool_get_lot_info(lot_id)
    if "error" in lot:
        return lot
    return {
        "lot": lot,
        "metrology": tool_get_lot_metrology(lot_id),
    }


def scenario_trace_recent_lots(eqp_id: str, limit: int = 5) -> dict[str, Any]:
    """[체이닝 + 팬아웃] 장비가 최근 진행한 랏 N개의 현재 위치/상태 추적.

    대표 질문: "4EKE0201 장비 상태랑, 최근에 진행한 5개 랏의 현재 위치 알려줘"

    이 시나리오가 하이브리드 구조의 핵심 예시다.
    - 랏 ID들은 질문에 없고, 1차 조회(장비) 결과 안에서 생겨난다.
      -> 태스크 추출 시점에는 존재하지 않아도 되는 "파생 값"
    - 몇 개가 나오든 반복문 + 스레드풀로 전부 조회한다.
      -> ReAct였다면 약한 모델이 N개 중 일부만 조회하고 넘어가거나
         호출 JSON을 깨뜨리는 지점. 코드는 이런 실수가 불가능하다.
    """
    # LLM이 추출한 파라미터는 신뢰하지 않고 코드에서 범위를 강제한다.
    # ("최근 100만개"라고 뽑아와도 시스템이 죽지 않게)
    limit = max(1, min(int(limit), 20))

    # --- 1차 조회: 장비 상태 -------------------------------------------------
    eqp = tool_get_equipment_status(eqp_id)
    if "error" in eqp:
        return eqp

    # --- 앞 결과에서 다음 조회의 인자를 "코드"가 꺼낸다 (체이닝의 핵심) ------
    # 현재 물고 있는 랏도 "진행한 랏"에 포함시킨다 (있으면 맨 앞에)
    lot_ids: list[str] = []
    if eqp.get("current_lot_id"):
        lot_ids.append(eqp["current_lot_id"])
    lot_ids.extend(eqp.get("recent_lot_ids", []))
    lot_ids = lot_ids[:limit]

    if not lot_ids:
        return {
            "equipment": eqp,
            "note": "이 장비의 진행 랏 이력이 없습니다.",
            "traced_lots": [],
        }

    # --- 2차 조회: 랏들의 현재 위치 (팬아웃, 병렬) ---------------------------
    # 각 랏 조회는 서로 독립이므로 스레드풀로 동시에 실행한다.
    with ThreadPoolExecutor(max_workers=min(len(lot_ids), _MAX_FANOUT_WORKERS)) as pool:
        lots = list(pool.map(tool_get_lot_info, lot_ids))

    return {
        "equipment": eqp,
        "requested_count": limit,
        "found_count": len(lot_ids),  # 요청보다 이력이 적으면 요약 LLM이 그 사실도 답함
        "traced_lots": lots,
    }


def scenario_diagnose_hold(lot_id: str) -> dict[str, Any]:
    """[체이닝] 랏이 왜 홀드됐는지 진단. (랏 조회 -> 계측 조회)

    대표 질문: "LOT-B001 왜 홀드됐어?"
    홀드 원인의 대부분은 계측 스펙 아웃이므로, 계측 이력에서
    FAIL 레코드를 코드가 미리 골라내 준다.
    """
    lot = tool_get_lot_info(lot_id)
    if "error" in lot:
        return lot

    metrology = tool_get_lot_metrology(lot_id)
    # FAIL 레코드만 미리 추려서 요약 LLM이 원인을 헷갈리지 않게 한다.
    fail_records = [
        r for r in metrology.get("records", []) if r.get("result") == "FAIL"
    ]

    return {
        "lot": lot,
        "is_hold": lot.get("state") == "HOLD",
        "hold_reason": lot.get("hold_reason"),
        "fail_metrology_records": fail_records,
        "all_metrology": metrology,
    }


def scenario_find_capable_equipment(process_step: str) -> dict[str, Any]:
    """[단일 조회] 특정 공정 스텝을 진행 가능한 장비 목록 + 즉시 배정 가능 여부."""
    return {"capable_equipment": tool_find_capable_equipment(process_step)}


def scenario_list_hold_lots() -> dict[str, Any]:
    """[단일 조회] 현재 홀드 상태인 랏 전체 목록."""
    return {"hold_lots": tool_search_lots(state="HOLD")}


# =============================================================================
# 시나리오 레지스트리
#
# 이 dict가 시스템의 "태스크 카탈로그"다. 세 곳에서 사용된다.
#   1) extraction.py: description/params로 태스크 추출 프롬프트를 자동 생성
#   2) extraction.py: params 스펙으로 추출 결과를 검증 (필수값, 엔티티 존재)
#   3) graph.py: task 이름으로 func을 찾아 실행
#
# params의 "entity" 값 의미 (extraction.py의 DB 검증에서 사용):
#   "eqp"  -> 장비 ID. DB(mes_mock)에 존재하는지 검증
#   "lot"  -> 랏 ID. DB에 존재하는지 검증
#   "step" -> 공정 스텝 이름. 검증 없이 통과 (툴이 힌트와 함께 에러 반환)
#   "int"  -> 숫자 파라미터. 시나리오 함수 내부에서 범위 클램프
# =============================================================================

SCENARIO_REGISTRY: dict[str, dict[str, Any]] = {
    "EQUIPMENT_STATUS": {
        "func": scenario_equipment_status,
        "description": "장비 1대의 현재 상태(가동/대기/고장/PM), 진행 중인 랏을 조회. 예: '4EKE0201 상태 알려줘'",
        "params": {
            "eqp_id": {"required": True, "entity": "eqp", "desc": "장비 ID"},
        },
    },
    "LOT_STATUS": {
        "func": scenario_lot_status,
        "description": "랏 1개의 현재 위치(공정 스텝/장비), 상태, 계측 이력을 조회. 예: 'LOT-A001 지금 어디 있어?'",
        "params": {
            "lot_id": {"required": True, "entity": "lot", "desc": "랏 ID"},
        },
    },
    "TRACE_RECENT_LOTS": {
        "func": scenario_trace_recent_lots,
        "description": (
            "장비 상태를 조회하고, 그 장비가 최근 진행한 랏 N개의 현재 위치/상태까지 "
            "한번에 추적. 예: '이 장비가 최근 진행한 5개 랏의 현재 위치 알려줘'"
        ),
        "params": {
            "eqp_id": {"required": True, "entity": "eqp", "desc": "장비 ID"},
            "limit": {"required": False, "entity": "int", "desc": "추적할 랏 개수 (기본 5)"},
        },
    },
    "DIAGNOSE_HOLD": {
        "func": scenario_diagnose_hold,
        "description": "랏이 홀드된 원인을 진단 (홀드 사유 + 계측 FAIL 레코드). 예: 'LOT-B001 왜 홀드됐어?'",
        "params": {
            "lot_id": {"required": True, "entity": "lot", "desc": "랏 ID"},
        },
    },
    "FIND_CAPABLE_EQUIPMENT": {
        "func": scenario_find_capable_equipment,
        "description": "특정 공정 스텝을 진행 가능한 장비 목록과 즉시 배정 가능 여부를 조회. 예: 'ETCH 어디서 할 수 있어?'",
        "params": {
            "process_step": {"required": True, "entity": "step", "desc": "공정 스텝 이름 (예: PHOTO, ETCH)"},
        },
    },
    "LIST_HOLD_LOTS": {
        "func": scenario_list_hold_lots,
        "description": "현재 홀드 상태인 랏 전체 목록을 조회. 예: '지금 홀드된 랏 다 보여줘'",
        "params": {},
    },
}


def run_scenario(task: dict[str, Any]) -> dict[str, Any]:
    """추출된 태스크 1건을 실행한다.

    Args:
        task: extraction.py가 만든 태스크 dict.
            {"task": "TRACE_RECENT_LOTS", "params": {"eqp_id": "...", "limit": 5}}

    Returns:
        시나리오 함수의 결과 dict. 실행 자체가 실패하면 {"error": ...}.
        (요약 LLM이 에러도 자연어로 설명할 수 있도록 항상 dict를 반환)
    """
    name = task.get("task", "")
    entry = SCENARIO_REGISTRY.get(name)
    if entry is None:
        # extraction.py가 검증하므로 정상 흐름에서는 도달하지 않지만,
        # 방어적으로 처리해 둔다.
        return {"error": f"알 수 없는 태스크: {name}"}

    func: Callable[..., dict[str, Any]] = entry["func"]
    try:
        return func(**task.get("params", {}))
    except Exception as e:
        # 실서버라면 MES API 장애 등이 여기 걸린다.
        return {"error": f"시나리오 실행 중 오류: {e}"}
