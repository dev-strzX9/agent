# 반도체 MES 조회 에이전트 - 노드형 ReAct + 추출 파이프라인

Tool calling 대신 **노드 + LLM 라우터**로 구현한 ReAct 에이전트입니다.
앞단에 **2단계 추출 파이프라인**(Pydantic 검증 + 조건부 보정 + DB 검증)을
두어, router가 "질의 체크리스트"를 기준으로 판단하게 합니다.

```
        START
          │
          ▼
       extract                       1차 추출 (LLM): 질문 → queries 체크리스트
          │                          Pydantic 검증 + 정규화 (대문자화, limit 클램프)
          ▼
       repair                        2차 보정+검증: 식별자 빠졌을 때만 보정 LLM,
          │                          추출된 ID의 DB 실존 확인 (환각/오타 필터)
          ▼
    ┌── router ─────────────────┐   판단: LLM이 queries vs results 대조
    │     ├─ fetch_equipment ───┤   실행: 장비 조회 (LLM 없음)
    │     ├─ fetch_lot ─────────┘   실행: 랏 조회/팬아웃 (LLM 없음)
    │     │                          실행 후 router로 복귀 = ReAct 루프
    │     └─ summarize → END         종료: 결과를 답변으로 정리 (LLM)
```

## 3중 방어 (안정성 장치)

| 단계 | 담당 | 하는 일 |
|---|---|---|
| 표기 정규화 | 코드 (Pydantic validator) | ID 대문자화/공백 제거, limit 1~20 클램프, 허용 외 needs 제거 |
| 식별자 보정 | LLM (조건부) | 1차 추출에서 ID가 빠진 query만 질문 재해석으로 보충 |
| 실존 확인 | 코드 (DB) | 없는 장비/랏 ID에 validation_error + 전체 목록 힌트 부착 |

추출이 실패해도 죽지 않습니다. `phase="extraction_failed"`만 기록하고
router가 질문 원문으로 동작하는 폴백 모드로 진행합니다.

## Tool calling ReAct와의 차이

| | Tool calling | 노드형 (이 프로젝트) |
|---|---|---|
| 판단 근거 | messages (대화 + tool 결과) | 구조화된 **state** (question + results) |
| 액션 표현 | `tool_calls` JSON (API 규격) | `{"next": "노드이름", "params": {...}}` (일반 JSON) |
| 실행 주체 | 우리 코드 | 우리 코드 (동일) |
| 팬아웃 | LLM이 tool_calls N건 생성해야 함 | 노드가 `lot_ids` 배열 받아 코드로 병렬 처리 |

체이닝(장비 조회 결과의 `recent_lot_ids` → 랏 조회)은 router(LLM)가
results 안의 ID를 읽고 다음 `params`에 넣는 방식으로 일어납니다.

## 실행 방법

```bash
cd react_agent
pip install -r requirements.txt

export LLM_API_KEY="..."
# export LLM_API_BASE="http://localhost:11434/v1"   # 사내 서버/Ollama 등
# export LLM_MODEL="glm-..."

python main.py
```

## 파일 구조 (읽는 순서 추천)

| 순서 | 파일 | 역할 |
|---|---|---|
| 1 | `config.py` | 접속 정보, 턴 상한 등 설정 |
| 2 | `mes/mock.py` | 가짜 MES 데이터. **실서버 전환 시 이 파일만 교체** |
| 3 | `schemas.py` | 추출 결과의 Pydantic 스키마 + 정규화 validator |
| 4 | `state.py` | 공유 상태 (question, queries, phase, results...) |
| 5 | `nodes/extract/node.py` | 1차 추출: 질문 → queries 체크리스트 (LLM) |
| 6 | `nodes/repair/node.py` | 2차 보정(조건부 LLM) + DB 실존 확인 |
| 7 | `nodes/router/node.py` | **핵심.** queries vs results 대조해 다음 노드 선택 (LLM) |
| 8 | `nodes/fetch_equipment/node.py` | 장비 조회 실행 노드 |
| 9 | `nodes/fetch_lot/node.py` | 랏 조회 실행 노드 (팬아웃 지원) |
| 10 | `nodes/summarize/node.py` | 최종 답변 생성 노드 |
| 11 | `graph.py` | 그래프 조립 (extract → repair → router ⇄ 실행 노드) |
| 12 | `main.py` | 대화형 CLI |

## 새 조회 노드 추가 방법

1. `nodes/새노드/` 폴더 생성, `node.py`에 노드 함수 작성
   (조회 결과를 `{"results": [{"node": ..., "params": ..., "result": ...}]}`로 반환)
2. `nodes/router/node.py`의 `NODE_CATALOG`에 이름/설명 추가
   (LLM 라우터의 선택지에 자동 반영)
3. `graph.py`에 `add_node` + 분기 매핑 + `router` 복귀 엣지 3줄 추가

## 안전장치

- router 턴 수 상한 (`MAX_ROUTER_TURNS`, 기본 8) → 초과 시 강제 summarize
- router 출력 JSON 파싱 실패 / 모르는 노드 이름 → 강제 summarize
- 조회 실패(없는 ID)도 에러를 results에 남김 → router가 정정 재시도하거나
  summarize가 "찾을 수 없음 + 전체 목록 힌트"로 답변
