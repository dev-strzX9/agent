"""
config.py - 환경 설정

LLM API 접속 정보와 에이전트 동작 파라미터.
환경 변수로 오버라이드할 수 있다.
"""

import os

# --- LLM API (OpenAI 호환 /chat/completions) ---------------------------------
LLM_API_BASE: str = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "")
LLM_MODEL: str = os.environ.get("LLM_MODEL", "gpt-4o")

REQUEST_TIMEOUT_SEC: int = 120

# 조회 에이전트는 정확성이 중요하므로 낮게.
LLM_TEMPERATURE: float = 0.1

# --- 루프 안전장치 ------------------------------------------------------------
# router(판단 노드)가 실행될 수 있는 최대 횟수.
# LLM 라우터가 같은 노드를 무한히 고르는 것을 막는다.
# 이 횟수에 도달하면 강제로 summarize로 보낸다.
MAX_ROUTER_TURNS: int = 8

# 랏 팬아웃(여러 랏 동시 조회) 시 동시 실행 상한. 실서버 MES 과부하 방지.
MAX_FANOUT_WORKERS: int = 8
