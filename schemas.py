"""
schemas.py - 추출 결과의 Pydantic 스키마 + 코드 레벨 정규화

ermap_agent.py의 패턴을 따른다. 핵심 철학:
  "LLM에게 형식을 맞춰 달라고 비는 대신, 어떤 형식으로 나와도 코드가 교정한다"

[여기서 하는 일]
1. LLM 추출 출력의 형태를 Pydantic 모델로 강제한다.
   - 필드 이름/타입이 틀리면 ValidationError로 즉시 드러난다.
   - 자유 JSON 파싱("첫 {부터 마지막 }까지")보다 훨씬 견고하다.
2. field_validator로 표기 변형을 정규화한다.
   - ID 앞뒤 공백 제거 + 대문자 통일 ("4eke0201 " -> "4EKE0201")
   - limit 범위 클램프 (1~20)
   - needs 값을 허용 목록으로 필터링 (LLM이 지어낸 need 제거)

[주의: 이 파일에서는 `from __future__ import annotations`를 쓰지 않는다]
Pydantic은 어노테이션을 런타임에 평가하는데, future import를 쓰면
"str | None" 같은 PEP 604 문자열이 파이썬 3.9에서 평가 불가라 죽는다.
그래서 ermap_agent와 동일하게 typing.Optional/List를 사용한다.
"""

from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator

# =============================================================================
# needs: "이 대상에 대해 무엇을 알고 싶은가"
#
# 태스크 이름(TRACE_RECENT_LOTS 같은 고정 카탈로그) 대신 needs 조합을 쓴다.
# 새 정보 유형이 생기면 여기 목록에 추가 + router의 NODE_CATALOG에
# 대응 노드를 추가하면 된다. (시나리오 함수를 만들 필요 없음)
# =============================================================================

ALLOWED_NEEDS = {
    "status",                 # 장비/랏의 현재 상태
    "recent_lots_location",   # 장비가 최근 진행한 랏들의 현재 위치 (체이닝 필요)
    "lot_status",             # 특정 랏의 위치/상태
}

# limit(추적 랏 개수) 기본값과 상한. LLM이 "100만개"를 뽑아와도 여기서 잘린다.
DEFAULT_LOT_LIMIT = 5
MAX_LOT_LIMIT = 20


def _normalize_id(value: Any) -> Any:
    """장비/랏 ID 정규화: 공백 제거 + 대문자 통일.

    사용자가 "4eke0201"이라고 쳐도 DB의 "4EKE0201"과 매칭되게 한다.
    (실서버에서 ID 체계에 소문자가 섞여 있다면 이 함수만 고치면 된다)
    """
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned.upper() if cleaned else None
    return value


class QueryTask(BaseModel):
    """추출된 질의 1건 = "누구에 대해(식별자) + 무엇을(needs)".

    식별자(eqp_id/lot_id)는 "질문에 명시된 것"만 채운다.
    조회 결과에서 파생되는 값(예: 장비의 recent_lot_ids)은 여기 없다.
    그건 router ⇄ 실행 노드 루프가 실행 중에 처리한다.
    """

    eqp_id: Optional[str] = Field(default=None, description="장비 ID (질문에 명시된 경우)")
    lot_id: Optional[str] = Field(default=None, description="랏 ID (질문에 명시된 경우)")
    needs: List[str] = Field(default_factory=list, description="알고 싶은 정보 유형 목록")
    lot_limit: Optional[int] = Field(
        default=None, description="recent_lots_location일 때 추적할 랏 개수"
    )

    @field_validator("eqp_id", "lot_id", mode="before")
    @classmethod
    def _norm_ids(cls, value: Any) -> Any:
        return _normalize_id(value)

    @field_validator("needs", mode="after")
    @classmethod
    def _filter_needs(cls, value: List[str]) -> List[str]:
        # LLM이 지어낸 need("wafer_map" 등)는 조용히 버린다.
        # 전부 버려져 빈 리스트가 되면 repair/검증 단계에서 걸린다.
        return [n for n in value if n in ALLOWED_NEEDS]

    @field_validator("lot_limit", mode="before")
    @classmethod
    def _clamp_limit(cls, value: Any) -> Any:
        if value is None:
            return None
        try:
            return max(1, min(int(value), MAX_LOT_LIMIT))
        except (TypeError, ValueError):
            return None

    def has_identifier(self) -> bool:
        """조회를 시작할 수 있는 식별자가 하나라도 있는가."""
        return bool(self.eqp_id or self.lot_id)


class ExtractedQueries(BaseModel):
    """1차 추출(stage-1)의 전체 출력."""

    queries: List[QueryTask] = Field(
        default_factory=list, description="질문에서 추출한 질의 목록"
    )


class RepairTask(BaseModel):
    """2차 보정(stage-2) 출력의 1건. 식별자 필드만 보정 대상이다.

    ermap_agent의 ErmapTaskIdentifierRepair에 해당.
    needs 등 나머지 필드는 1차 결과를 신뢰하고 건드리지 않는다.
    """

    eqp_id: Optional[str] = None
    lot_id: Optional[str] = None

    @field_validator("eqp_id", "lot_id", mode="before")
    @classmethod
    def _norm_ids(cls, value: Any) -> Any:
        return _normalize_id(value)


class RepairQueries(BaseModel):
    """2차 보정의 전체 출력. 1차 queries와 같은 개수/순서여야 한다."""

    queries: List[RepairTask] = Field(default_factory=list)
