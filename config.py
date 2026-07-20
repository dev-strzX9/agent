"""
config.py - 환경 설정 모듈

LLM API 접속 정보와 에이전트 동작 파라미터를 한곳에 모아둔 파일.
환경 변수로 오버라이드할 수 있게 해서, 코드 수정 없이
개발/운영 환경을 전환할 수 있도록 한다.

[사용하는 환경 변수]
- LLM_API_BASE : OpenAI 호환 API의 베이스 URL (기본: OpenAI 공식)
                 사내 프록시나 vLLM, Ollama 등을 쓸 경우 여기만 바꾸면 됨.
                 예) http://localhost:11434/v1  (Ollama)
- LLM_API_KEY  : API 키. 로컬 서버라면 아무 값이나 넣어도 되는 경우가 많음.
- LLM_MODEL    : 사용할 모델 이름. 툴 호출(function calling)을 지원하는
                 모델이어야 한다. (parallel tool calls 지원 모델이면 더 좋음)
"""

import os

# --- LLM API 접속 정보 -------------------------------------------------------
# OpenAI 호환(/chat/completions) 엔드포인트라면 어떤 서버든 동작한다.
LLM_API_BASE: str = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "")
LLM_MODEL: str = os.environ.get("LLM_MODEL", "gpt-4o")

# --- HTTP 요청 설정 ----------------------------------------------------------
# LLM 응답이 길어질 수 있으므로 타임아웃은 여유 있게 잡는다. (초 단위)
REQUEST_TIMEOUT_SEC: int = 120

# --- 에이전트 루프 안전장치 --------------------------------------------------
# LLM이 툴을 무한히 체이닝하는 것을 막기 위한 상한선.
# "에이전트 노드 실행 횟수" 기준이며, 이 횟수를 넘으면 그래프가 강제 종료된다.
# 랭그래프 자체에도 recursion_limit(기본 25)이 있지만,
# 우리 쪽에서 의미 있는 에러 메시지를 내기 위해 별도로 관리한다.
MAX_AGENT_TURNS: int = 10

# --- LLM 샘플링 파라미터 -----------------------------------------------------
# 조회형 에이전트는 창의성보다 정확성이 중요하므로 temperature를 낮게 둔다.
LLM_TEMPERATURE: float = 0.1

# --- 병렬 툴 호출 (ReAct 폴백 경로에서만 의미 있음) ---------------------------
# True면 모델이 한 턴에 여러 tool_calls를 반환할 수 있다.
# 회사 LLM(GLM 낮은 버전)처럼 약한 모델은 병렬 호출에서 인자를 섞거나
# JSON 형식을 깨는 경우가 많아 기본값을 False로 둔다. (한 턴에 1개씩이 안정적)
# 강한 모델로 바꾸면 True로 켜서 턴 수를 줄일 수 있다.
#
# 참고: 주 경로(시나리오)의 병렬성은 이 설정과 무관하다.
# 시나리오 팬아웃과 태스크 병렬 실행은 코드(ThreadPoolExecutor)가 담당한다.
PARALLEL_TOOL_CALLS: bool = os.environ.get("PARALLEL_TOOL_CALLS", "0") == "1"
