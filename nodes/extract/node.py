"""
nodes/extract - 1차 추출 노드 (stage-1): 질문 -> 질의 체크리스트

ermap_agent의 extract_entities_node에 해당. 질문이 들어오면 가장 먼저
실행되어, "누구에 대해 무엇을 알고 싶은지"를 구조화해 state["queries"]에
넣는다. 이 체크리스트가 이후 router의 판단 기준이 된다.

[extract를 앞에 두는 이유]
- router가 매 턴 질문 문장 전체를 재해석하는 대신,
  "체크리스트 중 뭐가 비었나"만 보면 되므로 판단 난이도가 내려간다.
- 여러 요구가 섞인 질문("A 상태+최근 랏, B 상태")이 query 단위로
  분리되어 로그/디버깅에서 진행 상황이 명확해진다.
- 추출 직후 식별자 정규화/DB 검증(repair 노드)이 가능해져,
  없는 장비를 조회하러 가는 낭비가 실행 전에 걸러진다.

[출력 형태는 Pydantic(schemas.ExtractedQueries)으로 강제]
chat_structured가 스키마 첨부 + 검증 + 정규화(대문자화, limit 클램프)를
수행한다. 실패하면 None -> phase="extraction_failed"로 기록하되,
그래프는 계속 진행한다. (router가 질문 원문만으로도 동작할 수 있으므로
추출 실패가 곧 전체 실패는 아니다)
"""

# 파이썬 3.9 호환 (Pydantic 모델이 없는 파일이므로 future import 사용 가능)
from __future__ import annotations

from typing import Any

from llm_client import chat_structured
from schemas import DEFAULT_LOT_LIMIT, ExtractedQueries
from state import AgentState

# few-shot 포함 추출 프롬프트.
# needs 값의 의미를 명확히 정의하고, "질문에 없는 값을 지어내지 말 것"을
# 예시로 보여준다. (약한 모델은 규칙 서술보다 예시 모방을 잘 따른다)
_SYSTEM_PROMPT = """\
당신은 반도체 MES 질문을 구조화하는 추출기입니다.
사용자 질문을 읽고, "누구에 대해(식별자) 무엇을(needs) 알고 싶은지"를
질의(query) 목록으로 추출하세요.

[needs 값의 의미]
- "status": 장비 또는 랏의 현재 상태
- "recent_lots_location": 그 장비가 최근 진행한 랏들의 현재 위치
  (이 need가 있으면 lot_limit도 함께 추출. 질문에 개수가 없으면 생략)
- "lot_status": 특정 랏의 위치/상태

[규칙]
1. 같은 대상(장비/랏)에 대한 요구는 하나의 query로 묶고 needs에 나열한다.
2. 다른 대상이면 별도 query로 분리한다.
3. 식별자(eqp_id/lot_id)는 질문에 명시된 것만 넣는다. 지어내지 않는다.
   (예: "최근 진행한 랏들"의 랏 ID는 질문에 없으므로 넣지 않는다.
    시스템이 장비 조회 후 알아서 처리한다)
4. 요구가 하나도 없으면 queries를 빈 배열로 출력한다.

[예시 1] 질문: "4EKE0201 상태랑 최근 진행한 5개 랏 위치 알려줘. 4EKE0202 상태도"
출력: {"queries": [
  {"eqp_id": "4EKE0201", "needs": ["status", "recent_lots_location"], "lot_limit": 5},
  {"eqp_id": "4EKE0202", "needs": ["status"]}
]}

[예시 2] 질문: "LOT-B001 지금 어디 있어?"
출력: {"queries": [{"lot_id": "LOT-B001", "needs": ["lot_status"]}]}

(주의: 예시의 ID는 예시일 뿐이다. 실제 출력에는 사용자 질문의 값만 쓸 것)
"""


def extract_node(state: AgentState) -> dict[str, Any]:
    """질문에서 질의 체크리스트를 추출한다. (LLM 1회, Pydantic 검증)"""
    parsed = chat_structured(
        system_prompt=_SYSTEM_PROMPT,
        user_content=state["question"],
        response_model=ExtractedQueries,
    )

    # --- 추출 실패: JSON이 깨졌거나 스키마 불일치 ----------------------------
    # 그래프는 계속 진행한다. queries가 비어 있으면 router가 질문 원문만
    # 보고 판단하는, extract 도입 전과 같은 모드로 동작한다.
    if parsed is None:
        return {
            "queries": [],
            "phase": "extraction_failed",
            "phase_message": "질의 추출 실패 (LLM 출력 파싱/검증 불가)",
        }

    queries = []
    for task in parsed.queries:
        q = task.model_dump(exclude_none=True)
        # recent_lots_location인데 개수가 없으면 기본값을 코드가 채운다.
        # (ermap의 _apply_default_dates와 같은 "기본값 규칙의 코드화")
        if "recent_lots_location" in q.get("needs", []) and "lot_limit" not in q:
            q["lot_limit"] = DEFAULT_LOT_LIMIT
        queries.append(q)

    return {
        "queries": queries,
        "phase": "extracted",
        "phase_message": f"질의 {len(queries)}건 추출 완료",
    }
