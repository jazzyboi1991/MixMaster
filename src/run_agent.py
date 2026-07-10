import sys

sys.path.append("src")

from agent import build_agent_graph


def run_test_session():
    app = build_agent_graph()

    # 동일한 thread_id를 설정하여 멀티턴을 테스트합니다.
    config = {"configurable": {"thread_id": "test_session_123"}}

    # 대화 1: BPM 딜레이 시간 계산 (도구 사용 테스트)
    query1 = "120 BPM 음악에서 1/8 박자 딜레이와 리버브 프리딜레이 값을 계산해줄래?"
    initial_state = {
        "query": query1,
        "category": "",
        "clarification_needed": False,
        "collected_details": {},
        "context_documents": [],
        "sources": [],
        "response": "",
        "steps": [],
        "grade_results": [],
        "rewrite_attempts": 0,
        "messages": [],
    }

    print(f"=== 대화 1 시작 ===\n질문: {query1}\n")
    for event in app.stream(initial_state, config=config):
        for node_name, state_update in event.items():
            print(f"▶ 실행된 노드: {node_name}")
            if "response" in state_update and state_update["response"]:
                print(f"💬 응답:\n{state_update['response']}\n")
            if "messages" in state_update:
                last_msg = state_update["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    print(f"🛠️ 도구 호출 감지: {last_msg.tool_calls}")
            print("-" * 50)

    # 대화 2: 이전 대화 템포 기억 검증 (멀티턴 테스트)
    query2 = "방금 계산해 준 세팅 중에서 1/16 박자 딜레이 값은 템포가 달라지면 어떻게 바뀌어? 예를 들어 60 BPM이면?"
    state2 = {
        "query": query2,
        "category": "",
        "clarification_needed": False,
        "collected_details": {},
        "context_documents": [],
        "sources": [],
        "response": "",
        "steps": [],
        "grade_results": [],
        "rewrite_attempts": 0,
        "messages": [],
    }

    print(f"\n=== 대화 2 시작 (멀티턴 검증) ===\n질문: {query2}\n")
    for event in app.stream(state2, config=config):
        for node_name, state_update in event.items():
            print(f"▶ 실행된 노드: {node_name}")
            if "response" in state_update and state_update["response"]:
                print(f"💬 응답:\n{state_update['response']}\n")
            if "messages" in state_update:
                last_msg = state_update["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    print(f"🛠️ 도구 호출 감지: {last_msg.tool_calls}")
            print("-" * 50)

    # 대화 3: 오디오 이론 지식 RAG 및 가드레일 예외 테스트
    query3 = "보컬 소리가 조금 먹먹해요. 어떻게 EQ를 만져야 하죠?"
    state3 = {
        "query": query3,
        "category": "",
        "clarification_needed": False,
        "collected_details": {},
        "context_documents": [],
        "sources": [],
        "response": "",
        "steps": [],
        "grade_results": [],
        "rewrite_attempts": 0,
        "messages": [],
    }

    print(f"\n=== 대화 3 시작 (RAG 툴 및 지식 답변 검증) ===\n질문: {query3}\n")
    for event in app.stream(state3, config=config):
        for node_name, state_update in event.items():
            print(f"▶ 실행된 노드: {node_name}")
            if "response" in state_update and state_update["response"]:
                print(f"💬 응답:\n{state_update['response']}\n")
            if "messages" in state_update:
                last_msg = state_update["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    print(f"🛠️ 도구 호출 감지: {last_msg.tool_calls}")
            print("-" * 50)


if __name__ == "__main__":
    run_test_session()
