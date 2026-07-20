"""
tools/mes_mock.py - 가짜 MES 데이터 계층 (Mock)

실제 환경에서는 이 파일이 MES REST API / DB 조회 코드로 교체된다.
그래서 일부러 "데이터 접근 함수"만 노출하고, 데이터 구조는 실제 MES 응답을
정규화한 형태(dict)로 맞춰두었다. 이 파일만 갈아끼우면 나머지 코드는
수정 없이 실서버에 붙는다.

[도메인 용어 정리]
- 랏(lot)        : 웨이퍼 묶음 단위. 공정 스텝을 순서대로 흘러간다.
- 공정 스텝(step): PHOTO, ETCH, DIFF(확산), CMP, METAL, 계측(METRO_*) 등.
- 계측(metrology): 공정 결과(두께, CD 등)를 측정하는 단계. 스펙 아웃이면
                   랏이 홀드(hold)될 수 있다.
- 장비 상태      : RUN(가동중) / IDLE(대기) / DOWN(고장) / PM(예방정비)

[체이닝을 위한 설계 포인트]
모든 반환값에 "다음 조회의 입력이 될 수 있는 ID"를 포함시킨다.
예) 장비 조회 결과에 recent_lot_ids가 있으므로,
    LLM이 그 ID로 get_lot_info를 이어서 호출할 수 있다.
"""

# 파이썬 3.9 호환: "str | None" 표기(PEP 604)를 쓰기 위한 지연 평가
from __future__ import annotations

from typing import Any

# =============================================================================
# 목업 데이터
# 실제로는 MES DB에 있을 내용. 시나리오 테스트가 가능하도록
# "DOWN 장비", "홀드 랏", "스펙 아웃 계측" 등을 골고루 심어두었다.
# =============================================================================

# 장비 마스터 + 실시간 상태
_EQUIPMENTS: dict[str, dict[str, Any]] = {
    "EQP-PHO-01": {
        "eqp_id": "EQP-PHO-01",
        "name": "Photo Track #1",
        "status": "RUN",              # 가동 중
        "current_lot_id": "LOT-A003", # 지금 물고 있는 랏
        "recent_lot_ids": ["LOT-A001", "LOT-A002"],  # 최근 완료한 랏 (최신순)
        "capable_steps": ["PHOTO"],   # 이 장비가 진행 가능한 공정 스텝
    },
    "EQP-PHO-02": {
        "eqp_id": "EQP-PHO-02",
        "name": "Photo Track #2",
        "status": "PM",               # 예방정비 중 -> 배정 불가
        "current_lot_id": None,
        "recent_lot_ids": ["LOT-B001"],
        "capable_steps": ["PHOTO"],
    },
    "EQP-ETC-01": {
        "eqp_id": "EQP-ETC-01",
        "name": "Etcher #1",
        "status": "IDLE",             # 대기 중 -> 즉시 배정 가능
        "current_lot_id": None,
        "recent_lot_ids": ["LOT-A001"],
        "capable_steps": ["ETCH"],
    },
    "EQP-ETC-02": {
        "eqp_id": "EQP-ETC-02",
        "name": "Etcher #2",
        "status": "DOWN",             # 고장 -> 배정 불가
        "current_lot_id": None,
        "recent_lot_ids": [],
        "capable_steps": ["ETCH"],
    },
    "EQP-MET-01": {
        "eqp_id": "EQP-MET-01",
        "name": "CD-SEM #1",
        "status": "RUN",
        "current_lot_id": "LOT-A002",
        "recent_lot_ids": ["LOT-A001", "LOT-B001"],
        "capable_steps": ["METRO_CD", "METRO_THK"],  # CD 계측, 두께 계측
    },
}

# 랏 현재 상태 (WIP: Work In Progress)
_LOTS: dict[str, dict[str, Any]] = {
    "LOT-A001": {
        "lot_id": "LOT-A001",
        "product": "DRAM-16G",
        "qty": 25,                    # 웨이퍼 매수
        "current_step": "ETCH",       # 현재 위치한 공정 스텝
        "current_eqp_id": None,       # 어느 장비에 있는지 (None이면 대기 중)
        "state": "WAIT",              # WAIT(스텝 대기) / PROC(진행 중) / HOLD(홀드)
        "hold_reason": None,
    },
    "LOT-A002": {
        "lot_id": "LOT-A002",
        "product": "DRAM-16G",
        "qty": 25,
        "current_step": "METRO_CD",
        "current_eqp_id": "EQP-MET-01",  # 계측 장비에서 진행 중
        "state": "PROC",
        "hold_reason": None,
    },
    "LOT-A003": {
        "lot_id": "LOT-A003",
        "product": "DRAM-16G",
        "qty": 24,
        "current_step": "PHOTO",
        "current_eqp_id": "EQP-PHO-01",
        "state": "PROC",
        "hold_reason": None,
    },
    "LOT-B001": {
        "lot_id": "LOT-B001",
        "product": "NAND-1T",
        "qty": 25,
        "current_step": "METRO_THK",
        "current_eqp_id": None,
        "state": "HOLD",              # 계측 스펙 아웃으로 홀드된 상태
        "hold_reason": "METRO_THK 스펙 아웃 (엔지니어 확인 대기)",
    },
}

# 랏별 계측 이력 (스텝별 측정 결과)
# result: PASS(스펙 인) / FAIL(스펙 아웃) / PENDING(측정 대기)
_METROLOGY: dict[str, list[dict[str, Any]]] = {
    "LOT-A001": [
        {
            "step": "METRO_CD",
            "result": "PASS",
            "measured_value": 45.2,
            "spec_min": 43.0,
            "spec_max": 47.0,
            "unit": "nm",
            "measured_at": "2026-07-19T10:12:00",
            "eqp_id": "EQP-MET-01",   # 어느 계측 장비에서 측정했는지
        },
    ],
    "LOT-A002": [
        {
            "step": "METRO_CD",
            "result": "PENDING",      # 현재 측정 진행 중
            "measured_value": None,
            "spec_min": 43.0,
            "spec_max": 47.0,
            "unit": "nm",
            "measured_at": None,
            "eqp_id": "EQP-MET-01",
        },
    ],
    "LOT-B001": [
        {
            "step": "METRO_THK",
            "result": "FAIL",         # 이 결과 때문에 랏이 HOLD 됨
            "measured_value": 1520.0,
            "spec_min": 1400.0,
            "spec_max": 1500.0,
            "unit": "A",
            "measured_at": "2026-07-20T08:30:00",
            "eqp_id": "EQP-MET-01",
        },
    ],
}


# =============================================================================
# 데이터 접근 함수
# 실서버 전환 시 이 함수들의 내부 구현만 requests 호출로 바꾸면 된다.
# =============================================================================

def fetch_equipment(eqp_id: str) -> dict[str, Any] | None:
    """장비 1대의 상태를 조회한다. 없으면 None."""
    return _EQUIPMENTS.get(eqp_id)


def fetch_lot(lot_id: str) -> dict[str, Any] | None:
    """랏 1개의 현재 상태(WIP)를 조회한다. 없으면 None."""
    return _LOTS.get(lot_id)


def fetch_metrology(lot_id: str, step: str | None = None) -> list[dict[str, Any]]:
    """랏의 계측 이력을 조회한다.

    Args:
        lot_id: 랏 ID
        step: 특정 계측 스텝만 필터링하고 싶을 때 지정 (예: "METRO_CD")
    """
    records = _METROLOGY.get(lot_id, [])
    if step:
        records = [r for r in records if r["step"] == step]
    return records


def fetch_equipments_by_step(process_step: str) -> list[dict[str, Any]]:
    """특정 공정 스텝을 진행 가능한 장비 목록을 조회한다.

    가용 여부 판단은 하지 않고 전체를 반환한다.
    (RUN/IDLE/DOWN 판단은 툴 계층에서 available 플래그로 표시)
    """
    return [
        eqp for eqp in _EQUIPMENTS.values()
        if process_step in eqp["capable_steps"]
    ]


def fetch_lots_filtered(
    eqp_id: str | None = None,
    step: str | None = None,
    state: str | None = None,
) -> list[dict[str, Any]]:
    """조건에 맞는 랏을 검색한다. 조건이 None이면 해당 조건은 무시."""
    results = []
    for lot in _LOTS.values():
        if eqp_id is not None and lot["current_eqp_id"] != eqp_id:
            continue
        if step is not None and lot["current_step"] != step:
            continue
        if state is not None and lot["state"] != state:
            continue
        results.append(lot)
    return results
