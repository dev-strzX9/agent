"""
nodes 패키지 - 그래프를 구성하는 노드들 (노드 1개 = 폴더 1개)

- router/          판단 노드. LLM이 state를 보고 다음 노드를 고른다. (ReAct의 Reason)
- fetch_equipment/ 실행 노드. 장비 조회.                              (ReAct의 Act)
- fetch_lot/       실행 노드. 랏 조회 (여러 개면 병렬 팬아웃).        (ReAct의 Act)
- summarize/       종료 노드. 조회 결과를 자연어 답변으로 정리.

새 조회 노드를 추가하려면:
  1) nodes/새노드/ 폴더를 만들고 node.py에 노드 함수 작성
  2) router/node.py의 NODE_CATALOG에 설명 추가 (LLM 선택지에 자동 반영)
  3) graph.py에 노드 등록 + 분기 매핑 1줄 추가
"""
