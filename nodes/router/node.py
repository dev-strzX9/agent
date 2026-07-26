"""
nodes/router - 판단 노드 (ReAct의 Reason 단계)

노드형 ReAct의 심장. 실행 노드가 끝날 때마다 여기로 돌아오고(복귀),
LLM이 state(질문 + 지금까지의 조회 결과)를 보고 다음 액션을 고른다.

[방법 2: LLM 기반 라우팅]
매 턴 LLM에게 아래를 통째로 보여준다.
  - 사용자 질문
  - 선택 가능한 노드 카탈로그 (이름/용도/파라미터)
  - 지금까지 실행한 조회와 그 결과 (state["results"])
그리고 "다음 노드 하나"를 JSON으로 고르게 한다.
    {"next": "fetch_lot", "params": {"lot_ids": ["LOT-A001"]}, "reason": "..."}

tool calling과 비교하면:
  - tool calling: LLM이 tool_calls(함수명+인자 JSON)를 API 규격으로 생성
  - 노드 라우팅:  LLM이 "다음 노드 이름 + 파라미터"를 일반 텍스트(JSON)로 출력
동작 원리는 같고(LLM이 실행할 것을 결정), 출력 형식과 실행 주체가 다르다.
체이닝(장비 결과의 recent_lot_ids -> 랏 조회)은 LLM이 results 안의
ID를 읽고 params에 넣는 방식으로 일어난다.

[안전장치]
  - JSON 파싱 실패 / 모르는 노드 이름 -> summarize로 강제 종료
    (조회가 하나도 없었어도 summarize가 "조회 실패" 답변을 만들 수 있다)
  - router_turns >= MAX_ROUTER_TURNS -> summarize로 강제 종료 (무한 루프 방지)
"""

# 파이썬 3.9 호환
from __future__ import annotations

import json
import re
from typing import Any, Literal

from config import MAX_ROUTER_TURNS
from llm_client import chat_completion
from state import AgentState

# =============================================================================
# 노드 카탈로그: LLM에게 보여줄 "선택지 메뉴판"
#
# 여기 설명이 곧 LLM의 선택 기준이 된다. 새 실행 노드를 만들면
# 이 dict에 추가하는 것만으로 router의 선택지에 반영된다.
# =============================================================================

NODE_CATALOG: dict[str, str] = {
    "fetch_equipment": (
        "장비 1대의 현재 상태를 조회한다. params: {\"eqp_id\": \"장비ID\"}. "
        "반환: 상태(RUN/IDLE/DOWN/PM), 현재 진행 랏(current_lot_id), "
        "최근 완료 랏 목록(recent_lot_ids, 최신순). "
        "반환된 랏 ID들은 fetch_lot의 lot_ids로 넣어 후속 조회할 수 있다."
    ),
    "fetch_lot": (
        "랏(웨이퍼 묶음)들의 현재 위치/상태를 조회한다. "
        "params: {\"lot_ids\": [\"랏ID\", ...]} (여러 개면 한 번에 넣을 것, 병렬 조회됨). "
        "반환: 현재 공정 스텝(current_step), 현재 장비(current_eqp_id), "
        "상태(WAIT/PROC/HOLD), 홀드 사유."
    ),
    "summarize": (
        "조회를 끝내고 최종 답변을 작성한다. params: {}. "
        "질문에 필요한 정보가 results에 전부 모였을 때 선택한다."
    ),
}

_SYSTEM_PROMPT = f"""\
당신은 반도체 MES 조회 에이전트의 라우터입니다.
질의 체크리스트(queries)와 지금까지의 조회 결과(results)를 대조해서,
다음에 실행할 노드 "하나"를 고르세요.

[선택 가능한 노드]
{chr(10).join(f"- {name}: {desc}" for name, desc in NODE_CATALOG.items())}

[규칙]
1. JSON만 출력한다. 형식: {{"next": "노드이름", "params": {{...}}, "reason": "한 줄 근거"}}
2. 체크리스트의 각 query를 위에서부터 확인하고, 아직 results로 충족되지
   않은 query의 needs를 채우는 노드를 고른다.
   - needs의 "status"/"recent_lots_location" (eqp_id 있음) -> 먼저 fetch_equipment
   - "recent_lots_location"은 장비 조회 결과의 recent_lot_ids로 fetch_lot까지 해야 충족
   - needs의 "lot_status" (lot_id 있음) -> fetch_lot
3. validation_error가 붙은 query는 조회하지 않는다 (이미 실패 확정,
   summarize가 힌트와 함께 설명한다).
4. params에 넣는 ID는 반드시 queries 또는 results에 실제로 등장한 값만
   사용한다. 지어내지 않는다.
5. 같은 조회를 반복하지 않는다. results에 이미 있는 조회는 다시 하지 않는다.
6. 모든 query가 충족되면(또는 남은 게 validation_error뿐이면) next=summarize.
7. 랏 여러 개는 fetch_lot 한 번에 lot_ids 배열로 모아서 요청한다.
8. 체크리스트가 비어 있으면([]), 사용자 질문을 직접 읽고 필요한 조회를
   스스로 판단한다. (추출 실패 시의 폴백 모드)

[예시]
queries: [{{"eqp_id": "EQP-PHO-01", "needs": ["status", "recent_lots_location"], "lot_limit": 5}}], results: []
-> {{"next": "fetch_equipment", "params": {{"eqp_id": "EQP-PHO-01"}}, "reason": "query 1의 장비 조회부터"}}

(위 조회 후) results에 recent_lot_ids=["LOT-A001","LOT-A002"]가 있음
-> {{"next": "fetch_lot", "params": {{"lot_ids": ["LOT-A001", "LOT-A002"]}}, "reason": "query 1의 recent_lots_location 충족을 위해 랏 조회 (lot_limit 이내)"}}

(랏 조회까지 끝남, 다른 query 없음)
-> {{"next": "summarize", "params": {{}}, "reason": "모든 query 충족"}}
"""


def _parse_router_output(text: str) -> dict[str, Any] | None:
    """LLM 출력에서 JSON을 관대하게 파싱한다.

    코드펜스(```json)나 앞뒤 설명이 붙어도 첫 '{'부터 마지막 '}'까지
    잘라 파싱을 시도한다. 실패하면 None (-> summarize로 강제 종료).
    """
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def router_node(state: AgentState) -> dict[str, Any]:
    """state를 LLM에게 보여주고 다음 노드를 결정한다.

    반환하는 상태 변경분:
        next_node   : 다음에 실행할 노드 이름
        next_params : 그 노드에 넘길 파라미터
        router_turns: +1 (누적)
    """
    # --- 안전장치: 턴 상한 도달 시 강제 종료 ---------------------------------
    if state.get("router_turns", 0) >= MAX_ROUTER_TURNS:
        return {
            "next_node": "summarize",
            "next_params": {},
            "router_turns": 1,
        }

    # --- LLM에게 state를 보여주고 다음 노드를 고르게 한다 --------------------
    # queries(체크리스트)와 results를 함께 준다.
    # - queries: extract/repair가 만든 "할 일 목록". router는 이것과 results를
    #   대조해 "어느 query의 어느 need가 아직 비었는지"만 판단하면 된다.
    #   (질문 문장 전체를 매 턴 재해석하는 것보다 판단 난이도가 낮다)
    # - results: 조회 결과 안의 ID(recent_lot_ids 등)를 LLM이 읽고
    #   다음 params에 쓸 수 있어야 체이닝이 된다.
    user_content = (
        f"[사용자 질문]\n{state['question']}\n\n"
        f"[질의 체크리스트 (queries)]\n"
        f"{json.dumps(state.get('queries', []), ensure_ascii=False, indent=2)}\n\n"
        f"[지금까지의 조회 결과 (results)]\n"
        f"{json.dumps(state.get('results', []), ensure_ascii=False, indent=2)}\n\n"
        f"다음 노드를 JSON으로 선택하세요."
    )
    reply = chat_completion([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ])

    parsed = _parse_router_output(reply)

    # --- 검증: 파싱 실패 또는 카탈로그에 없는 노드 -> summarize --------------
    # (여기서 죽는 것보다, 지금까지 모은 결과로라도 답변하는 게 낫다)
    if parsed is None or parsed.get("next") not in NODE_CATALOG:
        return {
            "next_node": "summarize",
            "next_params": {},
            "router_turns": 1,
        }

    return {
        "next_node": parsed["next"],
        "next_params": parsed.get("params", {}) or {},
        "router_turns": 1,
    }


def route_next(state: AgentState) -> Literal["fetch_equipment", "fetch_lot", "summarize"]:
    """랭그래프 조건 분기 함수. router_node가 정한 next_node로 라우팅한다."""
    return state["next_node"]  # type: ignore[return-value]
