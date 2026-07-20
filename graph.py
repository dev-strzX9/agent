"""
graph.py - 랭그래프 하이브리드 에이전트 정의

약한 LLM(GLM 낮은 버전) 환경을 위한 "태스크 추출 + 시나리오 실행"을
주 경로로, ReAct 루프를 폴백으로 가지는 하이브리드 구조.

[그래프 전체 구조]

                 START
                   |
                   v
              extract (LLM 1회: 태스크 분류 + 파라미터 추출 + DB 검증)
                   |
        분류 성공  |  분류 실패 (UNKNOWN / 파싱 실패)
       +-----------+---------------------+
       |                                 |
       v                                 v
   scenarios                        react_agent <--+
   (시나리오 함수 실행,                  |         |
    체이닝/팬아웃은 코드가 처리,         | tool_calls 있음
    태스크 여러 개면 병렬)               v         |
       |                            react_tools ---+
       v                                 |
   summarize                             | tool_calls 없음
   (LLM 1회: 결과를 자연어로)            v
       |                                END
       v
      END

[왜 이렇게 나눴나 - 각 경로에서 LLM이 하는 일]
- 주 경로(위 왼쪽): LLM 호출은 딱 2회.
    extract   = "보기 중에 고르기 + 질문에 있는 값 베끼기" (분류)
    summarize = "주어진 데이터를 문장으로 정리" (요약)
  약한 모델도 잘하는 작업만 시킨다. 틀리기 쉬운 판단
  (툴 선택, 호출 순서, 앞 결과에서 필드 꺼내기)은 전부
  시나리오 함수(파이썬 코드)가 수행하므로 모델 성능과 무관하게 정확하다.
- 폴백(위 오른쪽): 카탈로그에 없는 새로운 유형의 질문만 도달한다.
  성공률은 주 경로보다 낮지만 "처리 불가"로 끊는 것보다 낫다.
  여기로 자주 빠지는 유형이 보이면 tasks/scenarios.py에 시나리오로
  승격시켜 주 경로를 넓힌다. (폴백 = 새 태스크 후보 발견 채널)
"""

# 파이썬 3.9 호환: "str | None" 표기(PEP 604)를 쓰기 위한 지연 평가
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from config import MAX_AGENT_TURNS
from llm_client import chat_completion
from state import AgentState
from tasks.extraction import extract_tasks
from tasks.scenarios import run_scenario
from tools.definitions import TOOL_REGISTRY, TOOL_SCHEMAS


# =============================================================================
# 유틸: 대화 이력에서 필요한 조각 꺼내기
# =============================================================================

def _last_user_question(messages: list[dict[str, Any]]) -> str:
    """이력에서 가장 최근 user 메시지(이번 질문)를 찾는다."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content") or ""
    return ""


def _recent_context_text(messages: list[dict[str, Any]], max_turns: int = 2) -> str:
    """멀티턴 문맥용: 직전 user/assistant "텍스트" 메시지 몇 개를 요약 전달.

    "그럼 그 랏 계측은?" 같은 지시어("그 랏")를 태스크 추출이 해석하려면
    직전 대화가 필요하다. 툴 페이로드까지 다 주면 약한 모델이 헷갈리므로
    사람이 읽는 텍스트 턴만 골라서, 이번 질문 직전까지 최대 max_turns쌍을 준다.
    """
    texts = []
    # 마지막 메시지(이번 질문)는 제외하고 뒤에서부터 훑는다.
    for msg in reversed(messages[:-1]):
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            prefix = "사용자" if msg["role"] == "user" else "어시스턴트"
            texts.append(f"{prefix}: {msg['content']}")
        if len(texts) >= max_turns * 2:
            break
    return "\n".join(reversed(texts))


# =============================================================================
# 노드 1: extract - 태스크 추출 + 라우팅 결정 (주 경로의 입구)
# =============================================================================

def extract_node(state: AgentState) -> dict[str, Any]:
    """사용자 질문을 태스크 목록으로 분류하고 경로를 결정한다.

    실제 추출/검증 로직은 tasks/extraction.py에 있고,
    이 노드는 상태에서 질문을 꺼내 넘기고 결과를 상태에 싣는 역할만 한다.
    """
    question = _last_user_question(state["messages"])
    context = _recent_context_text(state["messages"])

    result = extract_tasks(question, context)

    if result["route"] == "react":
        # 폴백행. 사유는 로그/디버깅용으로 상태에 남긴다.
        return {"route": "react", "route_reason": result["reason"]}

    return {"route": "scenario", "tasks": result["tasks"]}


def route_after_extract(state: AgentState) -> Literal["scenarios", "react_agent"]:
    """extract 직후의 조건 분기: 주 경로 vs ReAct 폴백."""
    return "scenarios" if state["route"] == "scenario" else "react_agent"


# =============================================================================
# 노드 2: scenarios - 시나리오 함수 실행 (주 경로의 본체, LLM 개입 없음)
# =============================================================================

def scenarios_node(state: AgentState) -> dict[str, Any]:
    """추출된 태스크들을 실행한다.

    - 검증 실패 태스크(validation_error)는 실행하지 않고 에러 정보를
      결과로 넘긴다. "장비 없음"은 폴백이 아니라 답변으로 처리할
      정상 케이스이기 때문. (요약 LLM이 힌트와 함께 설명하게 됨)
    - 태스크가 여러 개면(다중 질문) 서로 독립이므로 병렬 실행한다.
      참고: 시나리오 "내부"의 팬아웃(랏 N개 조회)도 각자 병렬이므로,
      병렬이 2단으로 겹치는 구조다. 전부 I/O 대기(조회)라 스레드로 충분.
    """
    tasks = state["tasks"]

    def _run_one(task: dict[str, Any]) -> dict[str, Any]:
        if "validation_error" in task:
            # 실행 없이 검증 에러를 결과로 변환. hint(전체 장비 목록 등)가
            # 있으면 같이 실어서 요약 LLM이 정정 안내를 할 수 있게 한다.
            err: dict[str, Any] = {"error": task["validation_error"]}
            if "hint" in task:
                err["hint"] = task["hint"]
            return err
        return run_scenario(task)

    with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as pool:
        # pool.map은 입력 순서를 보존하므로 tasks[i] <-> results[i]가 대응된다.
        results = list(pool.map(_run_one, tasks))

    return {"scenario_results": results}


# =============================================================================
# 노드 3: summarize - 결과를 자연어 답변으로 (주 경로의 출구, LLM 1회)
# =============================================================================

_SUMMARIZE_SYSTEM_PROMPT = """\
당신은 반도체 팹 MES 조회 어시스턴트입니다.
시스템이 이미 필요한 데이터를 전부 조회해 두었습니다.
아래 조회 결과만을 근거로 사용자 질문에 한국어로 답하세요.

[규칙]
1. 조회 결과에 있는 수치/ID만 사용한다. 없는 정보를 지어내지 않는다.
2. 결과에 error가 있으면 그 이유를 설명하고, hint가 있으면
   (예: 전체 장비 목록) 사용자가 정정할 수 있게 안내한다.
3. 질문이 여러 개였으면 각각에 대해 구분해서 답한다.
4. 상태 코드는 풀어서 쓴다: RUN=가동중, IDLE=대기, DOWN=고장, PM=예방정비,
   WAIT=스텝 대기, PROC=진행중, HOLD=홀드, PASS=스펙 인, FAIL=스펙 아웃.
5. 간결하게, 근거 수치와 함께 답한다.
"""


def summarize_node(state: AgentState) -> dict[str, Any]:
    """시나리오 결과(JSON)를 요약 LLM에 넘겨 최종 답변을 생성한다.

    "주어진 데이터를 문장으로 정리"는 약한 모델도 잘하는 작업이다.
    결과에는 태스크 이름/파라미터를 함께 실어서, 어떤 질문에 대한
    데이터인지 모델이 매칭하기 쉽게 한다.
    """
    question = _last_user_question(state["messages"])

    # 태스크와 결과를 짝지어 하나의 구조로 만든다.
    bundle = [
        {
            "task": t.get("task"),
            "params": t.get("params", {}),
            "result": r,
        }
        for t, r in zip(state["tasks"], state["scenario_results"])
    ]

    messages = [
        {"role": "system", "content": _SUMMARIZE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"[사용자 질문]\n{question}\n\n"
                f"[조회 결과]\n{json.dumps(bundle, ensure_ascii=False, indent=2)}"
            ),
        },
    ]

    reply = chat_completion(messages, tools=None)

    # 최종 답변을 대화 이력에 assistant 메시지로 추가한다.
    # (다음 턴의 멀티턴 문맥으로도 쓰인다)
    return {"messages": [{"role": "assistant", "content": reply.get("content", "")}]}


# =============================================================================
# 노드 4, 5: ReAct 폴백 (react_agent <-> react_tools 순환)
#
# 카탈로그에 없는 질문만 도달하는 안전망. 동작 원리는 기존과 동일하다:
# LLM에게 툴 스키마를 주고, 툴 호출 -> 실행 -> 결과 보고 다음 판단 반복.
# 약한 모델 대비책:
#   - parallel_tool_calls는 config에서 기본 꺼져 있음 (한 턴에 1개씩)
#   - few-shot 예시는 main.py의 시스템 프롬프트에 포함
#   - 턴 수 상한(MAX_AGENT_TURNS)으로 무한 루프 차단
# =============================================================================

def react_agent_node(state: AgentState) -> dict[str, Any]:
    """[폴백] 대화 이력 전체를 LLM에 보내고 응답을 받는다.

    응답에 tool_calls가 있으면 툴을 쓰겠다는 뜻, 없으면 최종 답변이다.
    """
    # 안전장치: 턴 수 상한 도달 시 툴 없이 마무리 답변만 요청한다.
    # (툴 스키마를 계속 주면 또 호출하려 들 수 있으므로 tools=None으로 차단)
    if state.get("turn_count", 0) >= MAX_AGENT_TURNS:
        message = chat_completion(
            state["messages"]
            + [{
                "role": "user",
                "content": (
                    "툴 호출 횟수 상한에 도달했습니다. "
                    "지금까지 조회한 정보만으로 최종 답변을 작성해 주세요."
                ),
            }],
            tools=None,
        )
        return {"messages": [message], "turn_count": 1}

    message = chat_completion(state["messages"], tools=TOOL_SCHEMAS)
    return {"messages": [message], "turn_count": 1}


def _execute_single_tool(tool_call: dict[str, Any]) -> dict[str, Any]:
    """[폴백] tool_call 1건을 실행하고 role='tool' 메시지로 변환한다.

    OpenAI 포맷의 tool_call 구조:
        {
            "id": "call_abc123",              # 결과 매칭용 고유 ID
            "type": "function",
            "function": {
                "name": "get_lot_info",
                "arguments": "{\"lot_id\": \"LOT-A001\"}"  # JSON "문자열"임에 주의
            }
        }

    어떤 에러가 나든 예외를 던지지 않고 {"error": ...}를 결과로 넣는다.
    -> LLM이 에러를 보고 인자를 고쳐 재시도하거나, 사용자에게
       정상적으로 실패 사유를 답할 수 있게 하기 위함.
    또한 OpenAI 포맷 규칙상 tool_calls의 "모든" 호출에 각각 role='tool'
    응답이 있어야 다음 LLM 호출이 성공한다. (하나라도 빠지면 400 에러)
    """
    name = tool_call["function"]["name"]
    call_id = tool_call["id"]

    try:
        # arguments는 JSON 문자열이므로 dict로 파싱해야 한다.
        args = json.loads(tool_call["function"]["arguments"] or "{}")
    except json.JSONDecodeError as e:
        result: dict[str, Any] = {"error": f"툴 인자 JSON 파싱 실패: {e}"}
    else:
        func = TOOL_REGISTRY.get(name)
        if func is None:
            # LLM이 존재하지 않는 툴 이름을 지어낸 경우 (환각)
            result = {
                "error": f"알 수 없는 툴: {name}",
                "available_tools": list(TOOL_REGISTRY.keys()),
            }
        else:
            try:
                result = func(**args)
            except TypeError as e:
                # 필수 인자 누락, 잘못된 인자 이름 등
                result = {"error": f"툴 인자 오류: {e}"}
            except Exception as e:
                # 실서버라면 MES API 장애 등이 여기 걸린다.
                result = {"error": f"툴 실행 중 오류: {e}"}

    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(result, ensure_ascii=False),
    }


def react_tools_node(state: AgentState) -> dict[str, Any]:
    """[폴백] 직전 assistant 메시지의 tool_calls를 전부 실행한다.

    parallel_tool_calls가 꺼져 있으면(기본) 보통 1건씩 오지만,
    모델이 여러 건을 보내는 경우도 있으므로 병렬 실행을 유지한다.
    """
    last_message = state["messages"][-1]
    tool_calls = last_message.get("tool_calls", [])

    with ThreadPoolExecutor(max_workers=min(len(tool_calls), 8)) as pool:
        # pool.map은 입력 순서대로 결과를 반환 -> tool_calls와 순서 일치
        tool_messages = list(pool.map(_execute_single_tool, tool_calls))

    return {"messages": tool_messages}


def route_after_react_agent(state: AgentState) -> Literal["react_tools", "end"]:
    """[폴백] react_agent 직후 분기: 툴 실행 vs 종료(최종 답변)."""
    last_message = state["messages"][-1]
    if last_message.get("tool_calls"):
        return "react_tools"
    return "end"


# =============================================================================
# 그래프 조립
# =============================================================================

def build_graph():
    """하이브리드 에이전트 그래프를 조립하고 컴파일해서 반환한다.

    반환된 객체는 .invoke({"messages": [...], "turn_count": 0})로 실행한다.
    """
    builder = StateGraph(AgentState)

    # --- 노드 등록 ---
    builder.add_node("extract", extract_node)        # 태스크 추출 + 라우팅
    builder.add_node("scenarios", scenarios_node)    # 주 경로: 시나리오 실행
    builder.add_node("summarize", summarize_node)    # 주 경로: 답변 생성
    builder.add_node("react_agent", react_agent_node)  # 폴백: LLM 판단
    builder.add_node("react_tools", react_tools_node)  # 폴백: 툴 실행

    # --- 흐름 연결 ---
    builder.add_edge(START, "extract")

    # extract 후 분기: 주 경로 vs 폴백
    builder.add_conditional_edges(
        "extract",
        route_after_extract,
        {"scenarios": "scenarios", "react_agent": "react_agent"},
    )

    # 주 경로: 시나리오 실행 -> 요약 -> 종료 (직선)
    builder.add_edge("scenarios", "summarize")
    builder.add_edge("summarize", END)

    # 폴백: agent <-> tools 순환
    builder.add_conditional_edges(
        "react_agent",
        route_after_react_agent,
        {"react_tools": "react_tools", "end": END},
    )
    builder.add_edge("react_tools", "react_agent")

    return builder.compile()
