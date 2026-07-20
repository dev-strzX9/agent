"""
tasks/extraction.py - LLM 태스크 추출 + DB 엔티티 검증

하이브리드 구조의 입구. 사용자 질문을 받아 두 가지를 결정한다.
  1) 어떤 태스크(시나리오)들인가? + 질문에 명시된 파라미터는?
  2) 주 경로(시나리오)로 갈 수 있는가, ReAct 폴백으로 보내야 하는가?

[약한 모델(GLM 낮은 버전)을 위한 설계]
- LLM에게 시키는 일은 "보기 중에서 고르기 + 질문에 있는 값 베끼기"뿐.
  이건 약한 모델도 잘한다. 어려운 판단(툴 순서, 필드 추출)은
  전부 시나리오 함수(코드)가 가져갔다.
- function calling 대신 "JSON만 출력해" 방식을 쓴다.
  약한 모델은 툴 호출 포맷보다 순수 JSON 출력이 더 안정적인 경우가 많고,
  파싱 실패 시 우리가 직접 복구를 시도할 수 있다.
- few-shot 예시를 프롬프트에 포함한다. 약한 모델은 규칙 서술보다
  예시 모방을 훨씬 잘 따른다.

[여러 질문 동시 지원]
"4EKE0201 상태랑, 홀드된 랏도 다 보여줘"처럼 요구가 여러 개면
tasks 배열에 여러 건이 담긴다. 각 태스크는 독립적으로 실행된다(graph.py).

[DB 엔티티 검증]
LLM이 추출한 장비/랏 ID는 그대로 믿지 않고 DB(mes_mock)에 존재하는지
확인한다. 환각으로 지어낸 ID, 오타 ID가 여기서 걸러진다.
검증 실패한 태스크는 실행하지 않고 validation_error를 붙여서,
요약 LLM이 "해당 장비를 찾을 수 없다 (전체 장비: ...)"라고 답하게 한다.
"""

# 파이썬 3.9 호환: "str | None" 표기(PEP 604)를 쓰기 위한 지연 평가
from __future__ import annotations

import json
import re
from typing import Any

from llm_client import chat_completion
from tasks.scenarios import SCENARIO_REGISTRY
from tools import mes_mock


# =============================================================================
# 추출 프롬프트 생성
# =============================================================================

def _build_task_catalog_text() -> str:
    """SCENARIO_REGISTRY로부터 태스크 카탈로그 텍스트를 자동 생성한다.

    레지스트리에 시나리오를 추가하면 이 프롬프트에 자동 반영되므로,
    시나리오 추가 시 프롬프트를 따로 고칠 필요가 없다.
    """
    lines = []
    for name, entry in SCENARIO_REGISTRY.items():
        params_desc = ", ".join(
            f"{p}({'필수' if spec['required'] else '선택'}: {spec['desc']})"
            for p, spec in entry["params"].items()
        ) or "파라미터 없음"
        lines.append(f"- {name}: {entry['description']}\n  파라미터: {params_desc}")
    return "\n".join(lines)


# few-shot 예시: 약한 모델은 규칙보다 예시 모방을 잘 따르므로
# 대표 유형(단일/체이닝+파라미터/복수 태스크/미분류)을 하나씩 보여준다.
# 주의: 예시의 ID를 모델이 실제 답에 복사하는 실수를 막기 위해
# 마지막에 경고 문장을 넣어두었다.
_FEW_SHOT_EXAMPLES = """\
[예시 1] 질문: "EQP-PHO-01 상태 알려줘"
출력: {"tasks": [{"task": "EQUIPMENT_STATUS", "params": {"eqp_id": "EQP-PHO-01"}}]}

[예시 2] 질문: "EQP-MET-01이 최근에 진행한 3개 랏 지금 어디 있어?"
출력: {"tasks": [{"task": "TRACE_RECENT_LOTS", "params": {"eqp_id": "EQP-MET-01", "limit": 3}}]}

[예시 3] 질문: "EQP-ETC-01 상태랑 홀드된 랏 전부 보여줘"
출력: {"tasks": [{"task": "EQUIPMENT_STATUS", "params": {"eqp_id": "EQP-ETC-01"}}, {"task": "LIST_HOLD_LOTS", "params": {}}]}

[예시 4] 질문: "어제 PHOTO 지나간 랏 중에 계측 FAIL만 골라줘"
출력: {"tasks": [{"task": "UNKNOWN", "params": {}}]}

(주의: 위 예시의 ID는 예시일 뿐이다. 실제 출력에는 반드시 사용자 질문에 있는 값만 사용할 것)
"""


def _build_extraction_prompt() -> str:
    """태스크 추출용 시스템 프롬프트를 생성한다."""
    return f"""\
당신은 반도체 팹 MES 질문을 태스크로 분류하는 분류기입니다.
사용자 질문을 읽고, 아래 태스크 카탈로그에서 해당하는 태스크를 골라
JSON으로만 출력하세요. 설명이나 다른 텍스트는 절대 출력하지 마세요.

[태스크 카탈로그]
{_build_task_catalog_text()}

[규칙]
1. 출력 형식: {{"tasks": [{{"task": "태스크이름", "params": {{...}}}}]}}
2. 질문에 요구가 여러 개면 tasks 배열에 여러 건을 넣는다.
3. params에는 질문에 "명시된" 값만 넣는다. 질문에 없는 값을 지어내지 않는다.
   (예: 랏 ID가 질문에 없으면 넣지 않는다. 조회 결과에서 나올 값은
    시스템이 알아서 처리하므로 신경 쓰지 않는다.)
4. 카탈로그의 어떤 태스크에도 해당하지 않으면 {{"task": "UNKNOWN", "params": {{}}}}을 넣는다.
   억지로 비슷한 태스크에 끼워 맞추지 않는다.

{_FEW_SHOT_EXAMPLES}"""


# =============================================================================
# LLM 출력 파싱
# =============================================================================

def _parse_json_output(text: str) -> dict[str, Any] | None:
    """LLM 출력에서 JSON을 최대한 관대하게 파싱한다.

    약한 모델은 "JSON만 출력해"라고 해도
    - ```json ... ``` 코드펜스로 감싸거나
    - JSON 앞뒤에 설명 문장을 붙이는
    경우가 흔하다. 그래서 첫 '{'부터 마지막 '}'까지를 잘라내 파싱을
    시도한다. 그래도 실패하면 None을 반환한다 (-> ReAct 폴백행).
    """
    if not text:
        return None

    # 코드펜스 제거 (```json ... ``` 또는 ``` ... ```)
    text = re.sub(r"```(?:json)?", "", text).strip()

    # 첫 { 부터 마지막 } 까지 추출
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


# =============================================================================
# DB 엔티티 검증
# =============================================================================

def _validate_task(task: dict[str, Any]) -> dict[str, Any]:
    """태스크 1건을 검증하고, 문제가 있으면 validation_error를 붙인다.

    검증 항목:
      1. 태스크 이름이 레지스트리에 있는가 (UNKNOWN 포함 여부는 호출부에서 판단)
      2. 필수 파라미터가 전부 있는가
      3. 엔티티(장비/랏 ID)가 DB에 실제로 존재하는가  <- 환각/오타 필터
      4. 레지스트리에 정의되지 않은 파라미터 제거     <- 잘못된 인자로 함수 호출 방지

    검증 실패한 태스크는 시나리오를 실행하지 않고 에러 정보만 결과로 남긴다.
    (요약 LLM이 "장비 XXX를 찾을 수 없습니다. 전체 장비: ..."라고 답하게 됨)
    """
    name = task.get("task", "")
    entry = SCENARIO_REGISTRY.get(name)
    if entry is None:
        task["validation_error"] = f"알 수 없는 태스크 이름: {name}"
        return task

    raw_params = task.get("params", {}) or {}
    spec = entry["params"]

    # (4) 정의되지 않은 파라미터는 조용히 버린다.
    #     약한 모델이 스키마에 없는 키를 지어내도 시나리오 함수 호출이
    #     TypeError로 죽지 않게 하기 위함.
    params = {k: v for k, v in raw_params.items() if k in spec}
    task["params"] = params

    # (2) 필수 파라미터 확인
    for p, p_spec in spec.items():
        if p_spec["required"] and p not in params:
            task["validation_error"] = f"필수 파라미터 누락: {p} ({p_spec['desc']})"
            return task

    # (3) 엔티티 존재 검증 (DB 조회)
    #     LLM이 뽑은 ID를 그대로 믿지 않는다. 존재하지 않으면 실행 전에
    #     차단하고, 사용자가 정정할 수 있게 전체 목록을 힌트로 남긴다.
    for p, p_spec in spec.items():
        if p not in params:
            continue
        value = params[p]

        if p_spec["entity"] == "eqp" and mes_mock.fetch_equipment(value) is None:
            task["validation_error"] = f"장비 '{value}'가 DB에 존재하지 않습니다."
            task["hint"] = {"all_eqp_ids": [e["eqp_id"] for e in mes_mock._EQUIPMENTS.values()]}
            return task

        if p_spec["entity"] == "lot" and mes_mock.fetch_lot(value) is None:
            task["validation_error"] = f"랏 '{value}'가 DB에 존재하지 않습니다."
            return task

        # "step"은 여기서 검증하지 않는다. 툴 자체가 없는 스텝이면
        # 힌트와 함께 에러를 반환하므로 그쪽에 맡긴다.
        # "int"는 시나리오 함수 내부에서 범위를 클램프한다.

    return task


# =============================================================================
# 메인 진입 함수
# =============================================================================

def extract_tasks(question: str, context: str = "") -> dict[str, Any]:
    """사용자 질문에서 태스크 목록을 추출하고 검증한다.

    Args:
        question: 사용자의 이번 질문
        context: 직전 대화 요약 텍스트 (멀티턴에서 "그럼 그 랏은?" 같은
            지시어 해석을 돕기 위해 전달. 없으면 빈 문자열)

    Returns:
        {"route": "scenario", "tasks": [검증된 태스크, ...]}
        또는
        {"route": "react", "reason": "폴백 사유"}

    라우팅 정책:
        - 추출/파싱 실패          -> react (모델 출력이 깨짐)
        - 태스크 중 UNKNOWN 존재  -> react (카탈로그 밖의 질문)
          * 일부만 UNKNOWN이어도 전체를 폴백으로 보낸다. 질문을 쪼개서
            반만 시나리오로 답하면 사용자 입장에서 더 혼란스럽기 때문.
        - validation_error        -> scenario 유지 (실행은 안 하고
          에러를 요약 LLM에 전달. "장비 없음"은 폴백이 아니라 답변으로
          처리해야 할 정상 케이스이므로)
    """
    user_content = question
    if context:
        # 멀티턴 문맥: 직전 대화를 참고 정보로만 제공한다.
        user_content = f"[직전 대화 참고]\n{context}\n\n[이번 질문]\n{question}"

    messages = [
        {"role": "system", "content": _build_extraction_prompt()},
        {"role": "user", "content": user_content},
    ]

    # 툴 없이 순수 텍스트(JSON) 출력만 요청한다.
    reply = chat_completion(messages, tools=None)
    parsed = _parse_json_output(reply.get("content") or "")

    # --- 파싱 실패 -> ReAct 폴백 ---------------------------------------------
    if parsed is None or not isinstance(parsed.get("tasks"), list) or not parsed["tasks"]:
        return {"route": "react", "reason": "태스크 추출 결과를 파싱하지 못함"}

    tasks = parsed["tasks"]

    # --- UNKNOWN 포함 -> ReAct 폴백 ------------------------------------------
    if any(t.get("task") == "UNKNOWN" for t in tasks):
        return {"route": "react", "reason": "카탈로그에 없는 유형의 질문"}

    # --- 검증 (필수 파라미터, DB 엔티티 존재) --------------------------------
    validated = [_validate_task(t) for t in tasks]

    return {"route": "scenario", "tasks": validated}
