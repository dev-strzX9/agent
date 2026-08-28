"""
mes/mock.py - 가짜 MES 데이터 계층

실제 환경에서는 이 파일의 fetch_* 함수 내부를 MES REST API / DB 조회로
교체한다. 반환 dict 구조(필드명)를 유지하면 나머지 코드는 수정이 없다.

[도메인 용어]
- 랏(lot)   : 웨이퍼 묶음 단위. 공정 스텝을 순서대로 흘러간다.
- 장비 상태 : RUN(가동중) / IDLE(대기) / DOWN(고장) / PM(예방정비)
- 랏 상태   : WAIT(스텝 대기) / PROC(진행중) / HOLD(홀드)

[체이닝 포인트]
장비 조회 결과에 current_lot_id / recent_lot_ids 가 포함된다.
router(LLM)가 이 ID들을 보고 "랏 조회 노드"를 다음 액션으로 고를 수 있다.
"""

# 파이썬 3.9 호환
from __future__ import annotations

from typing import Any

_EQUIPMENTS: dict[str, dict[str, Any]] = {
    "EQP-PHO-01": {
        "eqp_id": "EQP-PHO-01",
        "name": "Photo Track #1",
        "status": "RUN",
        "current_lot_id": "LOT-A003",
        "recent_lot_ids": ["LOT-A001", "LOT-A002"],  # 최신순
        "capable_steps": ["PHOTO"],
    },
    "EQP-ETC-01": {
        "eqp_id": "EQP-ETC-01",
        "name": "Etcher #1",
        "status": "IDLE",
        "current_lot_id": None,
        "recent_lot_ids": ["LOT-A001"],
        "capable_steps": ["ETCH"],
    },
    "EQP-MET-01": {
        "eqp_id": "EQP-MET-01",
        "name": "CD-SEM #1",
        "status": "RUN",
        "current_lot_id": "LOT-A002",
        "recent_lot_ids": ["LOT-A001", "LOT-B001"],
        "capable_steps": ["METRO_CD", "METRO_THK"],
    },
}

_LOTS: dict[str, dict[str, Any]] = {
    "LOT-A001": {
        "lot_id": "LOT-A001",
        "product": "DRAM-16G",
        "qty": 25,
        "current_step": "ETCH",
        "current_eqp_id": None,      # None이면 스텝 대기 중
        "state": "WAIT",
        "hold_reason": None,
    },
    "LOT-A002": {
        "lot_id": "LOT-A002",
        "product": "DRAM-16G",
        "qty": 25,
        "current_step": "METRO_CD",
        "current_eqp_id": "EQP-MET-01",
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
        "state": "HOLD",
        "hold_reason": "METRO_THK 스펙 아웃 (엔지니어 확인 대기)",
    },
}


def fetch_equipment(eqp_id: str) -> dict[str, Any] | None:
    """장비 1대의 상태 조회. 없으면 None."""
    return _EQUIPMENTS.get(eqp_id)


def fetch_lot(lot_id: str) -> dict[str, Any] | None:
    """랏 1개의 현재 상태(WIP) 조회. 없으면 None."""
    return _LOTS.get(lot_id)


def all_equipment_ids() -> list[str]:
    """전체 장비 ID 목록 (에러 힌트용)."""
    return list(_EQUIPMENTS.keys())


def all_lot_ids() -> list[str]:
    """전체 랏 ID 목록 (에러 힌트용)."""
    return list(_LOTS.keys())
