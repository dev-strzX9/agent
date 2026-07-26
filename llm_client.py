"""
llm_client.py - requests 기반 LLM API 클라이언트

SDK 없이 requests로 OpenAI 호환 /chat/completions를 직접 호출한다.
이 프로젝트(노드형 ReAct)에서는 tool calling을 쓰지 않는다. LLM의 역할:
  1) extract/repair 노드에서 구조화된 JSON 출력 (chat_structured)
  2) router 노드에서 "다음 노드 이름" JSON 출력
  3) summarize 노드에서 최종 답변 텍스트 출력
"""

# 파이썬 3.9 호환: "str | None" 표기(PEP 604)를 쓰기 위한 지연 평가
from __future__ import annotations

import json
import re
from typing import Any, Optional, Type, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from config import (
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    REQUEST_TIMEOUT_SEC,
)


class LLMAPIError(Exception):
    """LLM API 호출 실패 (네트워크/HTTP/파싱)를 감싸는 예외."""


def chat_completion(messages: list[dict[str, Any]]) -> str:
    """대화를 보내고 assistant의 텍스트 응답(content)만 받아온다.

    Args:
        messages: OpenAI 포맷 메시지 리스트 (system/user/assistant).

    Returns:
        choices[0].message.content 문자열.

    Raises:
        LLMAPIError: 네트워크 오류, HTTP 에러, 응답 형식 오류 시.
    """
    url = f"{LLM_API_BASE.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SEC)
    except requests.RequestException as e:
        raise LLMAPIError(f"LLM API 요청 실패 (네트워크): {e}") from e

    if resp.status_code != 200:
        raise LLMAPIError(f"LLM API 오류 (HTTP {resp.status_code}): {resp.text[:500]}")

    try:
        return resp.json()["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError) as e:
        raise LLMAPIError(f"LLM 응답 파싱 실패: {resp.text[:500]}") from e


# =============================================================================
# 구조화 출력 (ermap_agent의 chat_structured 패턴)
# =============================================================================

M = TypeVar("M", bound=BaseModel)


def _extract_json_block(text: str) -> str | None:
    """LLM 출력에서 JSON 부분만 잘라낸다.

    약한 모델은 "JSON만 출력해"라고 해도 코드펜스(```json)나 설명 문장을
    붙이는 경우가 흔하다. 첫 '{'부터 마지막 '}'까지를 잘라 반환한다.
    """
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    return text[start : end + 1]


def chat_structured(
    system_prompt: str,
    user_content: str,
    response_model: Type[M],
) -> Optional[M]:
    """LLM 출력을 Pydantic 모델로 강제하는 구조화 호출.

    동작:
      1. 시스템 프롬프트에 response_model의 JSON 스키마를 자동으로 붙인다.
         (LLM이 필드 이름/타입을 볼 수 있게)
      2. 응답에서 JSON 블록을 잘라 model_validate로 검증한다.
         - Pydantic field_validator들이 이 시점에 실행되어
           ID 대문자화, limit 클램프 같은 정규화가 자동 적용된다.
      3. 파싱/검증 실패 시 None을 반환한다. (예외를 던지지 않음)
         호출부(extract 노드)가 None을 보고 실패 phase로 처리한다.

    Args:
        system_prompt: 역할/규칙/few-shot이 담긴 시스템 프롬프트
        user_content: 사용자 입력 (질문 등)
        response_model: 출력 형태를 정의한 Pydantic 모델 클래스

    Returns:
        검증된 모델 인스턴스, 실패 시 None.
    """
    schema = json.dumps(
        response_model.model_json_schema(), ensure_ascii=False, indent=2
    )
    full_system = (
        f"{system_prompt}\n\n"
        f"[출력 JSON 스키마]\n{schema}\n\n"
        f"위 스키마에 맞는 JSON만 출력하세요. 다른 텍스트는 출력하지 마세요."
    )

    reply = chat_completion([
        {"role": "system", "content": full_system},
        {"role": "user", "content": user_content},
    ])

    block = _extract_json_block(reply)
    if block is None:
        return None

    try:
        return response_model.model_validate(json.loads(block))
    except (json.JSONDecodeError, ValidationError):
        # 형식이 심하게 깨진 경우. 호출부에서 실패로 처리하게 한다.
        return None
