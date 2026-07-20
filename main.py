"""
main.py - 에이전트 실행 진입점 (대화형 CLI)

실행 방법:
    cd agent
    export LLM_API_KEY="..."          # 필요 시 LLM_API_BASE, LLM_MODEL도 설정
    python main.py

실행하면 대화형 프롬프트가 뜬다. 예시 질문:
    - EQP-PHO-01 상태 알려줘                                  <- 주 경로 (단일 조회)
    - EQP-MET-01이 최근에 진행한 2개 랏의 현재 위치 알려줘    <- 주 경로 (체이닝+팬아웃)
    - LOT-B001이 왜 홀드됐어?                                 <- 주 경로 (홀드 진단)
    - EQP-ETC-01 상태랑 홀드된 랏 전부 보여줘                 <- 주 경로 (다중 태스크)
    - PHOTO 진행중인 랏 중에 수량 24매짜리만 알려줘           <- 카탈로그 밖 -> ReAct 폴백

[이 CLI가 보여주는 것 (VERBOSE=True)]
질문마다 어느 경로로 라우팅됐는지, 시나리오/툴이 무엇을 반환했는지를
콘솔에 출력한다. 코드 흐름을 따라가며 볼 때 켜두면 좋다.
"""

# 파이썬 3.9 호환: 최신 타입 문법을 쓰기 위한 지연 평가
from __future__ import annotations

import json
import sys

from graph import build_graph
from llm_client import LLMAPIError

# =============================================================================
# 시스템 프롬프트 (ReAct 폴백 경로에서 사용)
#
# 주 경로(태스크 추출/요약)는 각자 전용 프롬프트를 쓰므로
# (tasks/extraction.py, graph.py 참고) 이 프롬프트는 사실상
# "카탈로그에 없는 질문이 폴백으로 왔을 때"의 행동 지침이다.
#
# 약한 모델(GLM 낮은 버전) 대비 포인트:
# - few-shot 예시 포함: 약한 모델은 규칙 서술보다 예시 모방을 잘 따른다.
#   특히 자주 틀리는 3가지(툴 이름, 인자 형식, 체이닝 시 필드 추출)를
#   예시로 직접 보여준다.
# - "상태는 항상 재조회" 규칙: 모델이 이전 턴 결과를 현재 상태로
#   단정하는 것을 막는다. (장비/랏 상태는 실시간으로 바뀜)
# =============================================================================
SYSTEM_PROMPT = """\
당신은 반도체 팹(FAB)의 MES 조회 어시스턴트입니다.
장비 상태, 랏(lot) 위치/상태, 계측(metrology) 결과, 공정별 가용 장비를
툴로 조회해서 엔지니어의 질문에 답합니다.

[행동 규칙]
1. 장비/랏/계측 상태는 반드시 툴로 조회한 최신 결과만 근거로 답하세요.
   이전 대화에서 조회했던 결과를 현재 상태로 단정하지 마세요.
2. 한 번에 툴을 하나씩 호출하고, 결과를 확인한 뒤 다음 툴을 호출하세요.
3. 후속 조회가 필요하면(예: 장비 조회 결과의 랏 ID로 랏 위치 확인)
   주저하지 말고 툴을 이어서 호출하세요.
4. 툴이 error를 반환하면 힌트를 참고해 정정 후 재시도하고,
   그래도 실패하면 사용자에게 이유를 설명하세요. 데이터를 지어내지 마세요.
5. 답변은 한국어로, 조회한 수치/ID를 근거로 간결하게 작성하세요.

[도메인 참고]
- 장비 상태: RUN(가동중), IDLE(대기), DOWN(고장), PM(예방정비)
- 랏 상태: WAIT(스텝 대기), PROC(진행중), HOLD(홀드)
- 계측 결과: PASS(스펙 인), FAIL(스펙 아웃), PENDING(측정 대기)

[올바른 진행 예시]
예시 A) 질문: "EQP-PHO-01 상태 알려줘"
  1턴: get_equipment_status(eqp_id="EQP-PHO-01") 호출
  2턴: 결과를 근거로 답변 작성

예시 B) 질문: "EQP-MET-01이 최근에 진행한 랏이 지금 어디 있어?"
  1턴: get_equipment_status(eqp_id="EQP-MET-01") 호출
  2턴: 결과의 recent_lot_ids에서 랏 ID를 꺼내
       get_lot_info(lot_id=<그 값>) 호출
  3턴: 두 결과를 근거로 답변 작성

예시 C) 질문: "LOT-B001이 왜 홀드됐어?"
  1턴: get_lot_info(lot_id="LOT-B001") 호출 -> state가 HOLD임을 확인
  2턴: get_lot_metrology(lot_id="LOT-B001") 호출 -> FAIL 레코드 확인
  3턴: 홀드 사유와 측정값/스펙을 근거로 답변 작성

(주의: 예시의 ID는 예시일 뿐입니다. 실제 조회에는 사용자 질문과
조회 결과에 나온 값만 사용하세요.)
"""

# 실행 과정(라우팅, 시나리오 결과, 툴 호출)을 콘솔에 보여줄지 여부.
VERBOSE = True


def print_step_trace(chunk: dict) -> None:
    """그래프 실행 중간 과정을 사람이 읽기 좋게 출력한다.

    graph.stream(..., stream_mode="updates")은 노드가 하나 실행될 때마다
    {노드이름: 상태변경분} 형태의 chunk를 내보낸다.
    노드별로 관심 있는 변경분을 골라 보여준다.
    """
    for node_name, update in chunk.items():
        if update is None:
            continue

        # --- extract 노드: 라우팅 결정과 추출된 태스크 ---
        if node_name == "extract":
            if update.get("route") == "react":
                print(f"  [라우팅] ReAct 폴백으로 이동 (사유: {update.get('route_reason')})")
            else:
                print("  [라우팅] 주 경로 (시나리오)")
                for t in update.get("tasks", []):
                    mark = " (검증 실패)" if "validation_error" in t else ""
                    print(f"  [태스크] {t.get('task')} params={t.get('params')}{mark}")

        # --- scenarios 노드: 시나리오 실행 결과 미리보기 ---
        elif node_name == "scenarios":
            for r in update.get("scenario_results", []):
                preview = json.dumps(r, ensure_ascii=False)
                if len(preview) > 200:
                    preview = preview[:200] + "..."
                print(f"  [시나리오 결과] {preview}")

        # --- ReAct 노드들: 툴 호출/결과 ---
        for msg in update.get("messages", []):
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc["function"]
                    print(f"  [툴 호출] {fn['name']}({fn['arguments']})")
            elif role == "tool":
                content = msg["content"]
                preview = content if len(content) <= 200 else content[:200] + "..."
                print(f"  [툴 결과] {preview}")


def run_once(app, history: list[dict], user_input: str) -> list[dict]:
    """사용자 입력 1건을 처리하고, 갱신된 대화 이력을 반환한다.

    Args:
        app: 컴파일된 랭그래프 앱
        history: 지금까지의 대화 이력 (OpenAI 포맷 dict 리스트)
        user_input: 사용자의 새 질문

    Returns:
        에이전트의 답변까지 포함된 새 대화 이력.
        (다음 질문에서 문맥으로 사용된다 -> 멀티턴 대화 지원.
         태스크 추출도 직전 대화를 참고하므로 "그럼 그 랏은?" 같은
         후속 질문이 어느 정도 동작한다)
    """
    history = history + [{"role": "user", "content": user_input}]

    # turn_count(ReAct 턴 수)는 매 질문마다 0에서 다시 시작한다.
    initial_state = {"messages": history, "turn_count": 0}

    if VERBOSE:
        # stream: 노드 실행 단위로 중간 과정을 받아본다.
        # 대화 이력은 각 노드가 반환한 messages 변경분을 직접 누적한다.
        for chunk in app.stream(initial_state, stream_mode="updates"):
            print_step_trace(chunk)
            for update in chunk.values():
                if update:
                    history = history + update.get("messages", [])
        final_messages = history
    else:
        final_state = app.invoke(initial_state)
        final_messages = final_state["messages"]

    # 마지막 assistant 메시지가 최종 답변이다.
    answer = final_messages[-1].get("content", "(응답 없음)")
    print(f"\n[답변]\n{answer}\n")

    return final_messages


def main() -> None:
    """대화형 CLI 메인 루프."""
    app = build_graph()

    # 대화 이력은 시스템 프롬프트로 시작한다.
    # (주 경로에서는 참고용, ReAct 폴백에서는 행동 지침으로 쓰인다)
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("=" * 60)
    print("반도체 MES 조회 에이전트 - 하이브리드 (종료: quit 또는 Ctrl+C)")
    print("=" * 60)
    print("예시 질문:")
    print("  - EQP-MET-01이 최근에 진행한 2개 랏의 현재 위치 알려줘")
    print("  - EQP-ETC-01 상태랑 홀드된 랏 전부 보여줘")
    print("  - LOT-B001이 왜 홀드됐어?")
    print()

    while True:
        try:
            user_input = input("질문> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n종료합니다.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("종료합니다.")
            break

        try:
            history = run_once(app, history, user_input)
        except LLMAPIError as e:
            # LLM API 문제(키 오류, 네트워크 등)는 프로그램을 죽이지 않고
            # 원인을 보여준 뒤 다음 질문을 받는다.
            print(f"\n[오류] {e}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
