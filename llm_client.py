"""
llm_client.py - requests 기반 LLM API 클라이언트

OpenAI SDK 같은 라이브러리 없이, requests로 직접
OpenAI 호환 /chat/completions 엔드포인트를 호출하는 얇은 클라이언트.

[왜 SDK 대신 requests인가]
- 의존성이 가벼워지고, 요청/응답 페이로드가 그대로 보여서 디버깅이 쉽다.
- 사내 프록시, vLLM, Ollama 등 OpenAI 호환 서버로 갈아탈 때 코드 수정이 없다.
- 에이전트 루프(graph.py)가 메시지를 "OpenAI 포맷 dict 그대로" 다루므로
  랭체인 메시지 객체 <-> dict 변환 비용도 없다.

[이 파일이 아는 것 / 모르는 것]
- 아는 것: HTTP 요청을 만들고, 응답에서 assistant 메시지를 꺼내는 방법.
- 모르는 것: 어떤 툴이 있는지, 대화가 어떻게 흘러가는지.
  -> 툴 정의는 tools/definitions.py, 루프 제어는 graph.py 책임.
"""

# 파이썬 3.9 호환: "str | None" 같은 타입 표기(PEP 604)는 3.10+ 문법이므로,
# 3.9에서도 동작하도록 어노테이션을 문자열로 지연 평가시킨다.
from __future__ import annotations

from typing import Any

import requests

from config import (
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    PARALLEL_TOOL_CALLS,
    REQUEST_TIMEOUT_SEC,
)


class LLMAPIError(Exception):
    """LLM API 호출 실패를 나타내는 예외.

    HTTP 에러(4xx/5xx), 타임아웃, 응답 파싱 실패를 모두 이 예외로 감싼다.
    호출부(graph.py)에서는 이 예외 하나만 처리하면 된다.
    """


def chat_completion(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """LLM에 대화 이력을 보내고 assistant 메시지 1개를 받아온다.

    Args:
        messages: OpenAI 포맷의 메시지 리스트. 각 원소는 다음 형태 중 하나.
            {"role": "system",    "content": "..."}
            {"role": "user",      "content": "..."}
            {"role": "assistant", "content": "...", "tool_calls": [...]}  # 툴 호출 포함 가능
            {"role": "tool",      "tool_call_id": "...", "content": "..."}  # 툴 실행 결과
        tools: OpenAI function calling 포맷의 툴 스키마 리스트.
            None이면 툴 없이 순수 텍스트 응답만 요청한다.

    Returns:
        응답의 choices[0].message (dict).
        - 툴을 호출하려는 경우: "tool_calls" 키에 호출 목록이 들어있다.
          모델이 병렬 툴 호출을 지원하면 tool_calls에 여러 개가 한번에 담긴다.
        - 최종 답변인 경우: "content"에 텍스트만 있고 tool_calls는 없다.

    Raises:
        LLMAPIError: 네트워크 오류, HTTP 에러, 응답 형식 오류 시.
    """
    url = f"{LLM_API_BASE.rstrip('/')}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        # 로컬 서버(Ollama 등)는 키가 없어도 되지만, 헤더는 항상 보내도 무방하다.
        "Authorization": f"Bearer {LLM_API_KEY}",
    }

    payload: dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
    }

    # 툴이 있을 때만 관련 필드를 넣는다.
    # tool_choice="auto": 툴을 쓸지 말지 모델이 스스로 판단.
    # (강제로 특정 툴을 쓰게 하려면 {"type": "function", "function": {"name": ...}} 형태)
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        # 한 턴에 여러 툴을 동시에 호출할 수 있게 허용할지 여부.
        # 약한 모델(GLM 낮은 버전 등)은 병렬 호출에서 실수가 잦아
        # 기본 False(한 턴에 1개씩)로 둔다. 자세한 설명은 config.py 참고.
        payload["parallel_tool_calls"] = PARALLEL_TOOL_CALLS

    try:
        resp = requests.post(
            url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SEC
        )
    except requests.RequestException as e:
        # 연결 실패, 타임아웃 등 네트워크 계층의 문제
        raise LLMAPIError(f"LLM API 요청 실패 (네트워크): {e}") from e

    if resp.status_code != 200:
        # 본문에 에러 원인이 들어있는 경우가 많으므로 함께 보여준다.
        # (예: 잘못된 API 키, 존재하지 않는 모델, 컨텍스트 초과 등)
        raise LLMAPIError(
            f"LLM API 오류 (HTTP {resp.status_code}): {resp.text[:500]}"
        )

    try:
        data = resp.json()
        message = data["choices"][0]["message"]
    except (ValueError, KeyError, IndexError) as e:
        raise LLMAPIError(f"LLM 응답 파싱 실패: {resp.text[:500]}") from e

    return message
