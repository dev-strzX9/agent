"""
state.py - 랭그래프 상태 정의

그래프의 모든 노드가 공유하는 상태(State)를 정의한다.

[설계 결정: 랭체인 메시지 객체 대신 dict를 쓰는 이유]
랭그래프는 보통 langchain_core의 AIMessage/HumanMessage 객체와
add_messages 리듀서를 함께 쓰지만, 우리는 LLM을 requests로 직접
호출하므로 API가 주고받는 "OpenAI 포맷 dict"를 그대로 상태에 쌓는다.
- 변환 계층이 없어 디버깅 시 페이로드가 그대로 보인다.
- llm_client.py의 입출력을 가공 없이 바로 상태에 넣을 수 있다.

[리듀서(Annotated + operator.add)란]
랭그래프에서 노드는 "상태 전체"가 아니라 "변경분"을 반환한다.
- Annotated[..., operator.add] 필드: 노드가 반환한 값이 기존 값에
  "더해진다" (리스트면 append, 숫자면 +).
- 어노테이션이 없는 필드: 노드가 반환하면 통째로 "덮어써진다".
  route/tasks/scenario_results는 질문마다 새로 계산되는 값이므로
  덮어쓰기가 맞다.
"""

# 파이썬 3.9 호환: 최신 타입 문법을 쓰기 위한 지연 평가
from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """에이전트 그래프의 공유 상태.

    total=False: 모든 키가 필수는 아니다. 예를 들어 시나리오 경로에서는
    turn_count가 쓰이지 않고, ReAct 경로에서는 tasks가 쓰이지 않는다.

    Attributes:
        messages: OpenAI 포맷의 대화 이력. (append-only)
            시나리오 경로: [system, user, ..., assistant(최종답변)]
            ReAct 경로:   [system, user, ..., assistant(tool_calls), tool, ..., assistant]
        turn_count: ReAct 폴백에서 LLM(agent) 노드가 실행된 횟수.
            무한 툴 체이닝 방지(config.MAX_AGENT_TURNS)에 사용. (누적)
        route: 태스크 추출 노드가 결정한 경로. "scenario" 또는 "react". (덮어쓰기)
        route_reason: 폴백으로 간 경우 그 사유 (디버깅/로그용). (덮어쓰기)
        tasks: 추출+검증된 태스크 목록. (덮어쓰기)
            [{"task": "TRACE_RECENT_LOTS", "params": {...}}, ...]
            검증 실패 태스크에는 validation_error 키가 붙어 있다.
        scenario_results: 태스크별 실행 결과. tasks와 인덱스가 대응된다. (덮어쓰기)
    """

    messages: Annotated[list[dict[str, Any]], operator.add]
    turn_count: Annotated[int, operator.add]
    route: str
    route_reason: str
    tasks: list[dict[str, Any]]
    scenario_results: list[dict[str, Any]]
