"""
nodes/summarize - 종료 노드: 최종 답변 생성 (LLM 1회)

router가 "필요한 조회가 다 끝났다"고 판단하면 여기로 온다.
state["results"]에 쌓인 조회 결과 전체를 LLM에게 주고,
사용자 질문에 대한 자연어 답변을 작성하게 한다.

"주어진 데이터를 문장으로 정리"는 라우팅보다 훨씬 쉬운 작업이라
모델 성능에 크게 좌우되지 않는다.
"""

# 파이썬 3.9 호환
from __future__ import annotations

import json
from typing import Any

from llm_client import chat_completion
from state import AgentState

_SYSTEM_PROMPT = """\
당신은 반도체 팹 MES 조회 어시스턴트입니다.
시스템이 이미 필요한 데이터를 조회해 두었습니다.
아래 조회 결과만을 근거로 사용자 질문에 한국어로 답하세요.

[규칙]
1. 조회 결과에 있는 수치/ID만 사용한다. 없는 정보를 지어내지 않는다.
2. 결과에 error가 있으면 그 이유를 설명하고, hint가 있으면
   (예: 전체 장비 목록) 사용자가 정정할 수 있게 안내한다.
3. 질문이 여러 개였으면 각각 구분해서 답한다.
4. 상태 코드는 풀어서 쓴다: RUN=가동중, IDLE=대기, DOWN=고장, PM=예방정비,
   WAIT=스텝 대기, PROC=진행중, HOLD=홀드.
5. 간결하게, 근거 수치와 함께 답한다.
6. 조회 결과가 비어 있으면 어떤 정보를 찾지 못했는지 솔직하게 말한다.
"""


def summarize_node(state: AgentState) -> dict[str, Any]:
    """results 전체를 근거로 최종 답변을 생성해 state["answer"]에 넣는다.

    queries(체크리스트)도 함께 준다. DB 검증에 실패한 질의의
    validation_error/hint가 여기 들어 있어서, "장비 XXX는 존재하지
    않습니다. 전체 장비: ..." 같은 정정 안내를 답변에 포함할 수 있다.
    """
    user_content = (
        f"[사용자 질문]\n{state['question']}\n\n"
        f"[질의 체크리스트 (검증 에러 포함 가능)]\n"
        f"{json.dumps(state.get('queries', []), ensure_ascii=False, indent=2)}\n\n"
        f"[조회 결과]\n"
        f"{json.dumps(state.get('results', []), ensure_ascii=False, indent=2)}"
    )
    answer = chat_completion([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ])
    return {"answer": answer}
