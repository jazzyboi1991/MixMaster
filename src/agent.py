from typing import Literal

from config import Config
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from retriever import Chroma, GoogleGenerativeAIEmbeddings
from state import AgentState


class QueryAnalysis(BaseModel):
    category: str = Field(
        description="일반 대화(general), 믹싱 이론(theory), 음향 문제 진단(troubleshooting) 중 하나"
    )
    clarification_needed: bool = Field(
        description="사용자의 질문에 정보가 모자라 추가 질문을 던져야 하면 True, 아니면 False"
    )
    instrument: str = Field(
        description="대상 악기 (예: Vocal, Drum, Bass 등), 없을 경우 'None'"
    )
    problem_type: str = Field(
        description="발생한 문제 유형 (예: Muddy, Harsh, Dynamics 등), 없을 경우 'None'"
    )


class GradeDocument(BaseModel):
    binary_score: str = Field(
        description="검색된 문서 조각이 사용자의 음향 질문에 실질적인 관련 지식(키워드, 원리)을 담고 있으면 'yes', 관련 없다면 'no'"
    )


@tool
def search_audio_knowledge(query: str) -> str:
    """오디오 믹싱, 마스터링, 이퀄라이저(EQ), 컴프레서, 리버브, 딜레이 등 음향 이론 및 장비 관련 지식을 검색합니다.
    음향 문제 해결이나 이론적인 지식, 장비 사용법이 필요할 때 이 도구를 사용하세요.
    """
    Config.validate()
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2", google_api_key=Config.GEMINI_API_KEY
    )
    db = Chroma(persist_directory=Config.DB_DIR, embedding_function=embeddings)

    docs = db.similarity_search(query, k=3)
    results = []
    for doc in docs:
        results.append(
            f"[출처: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
        )

    return (
        "\n\n---\n\n".join(results)
        if results
        else "관련 음향 지식을 데이터베이스에서 찾을 수 없습니다."
    )


@tool
def calculate_delay_time(bpm: float) -> str:
    """입력받은 BPM(템포)을 기준으로 음악의 박자별 딜레이 시간(ms) 및 추천 프리딜레이(pre-delay) 시간을 계산합니다.
    딜레이 시간 설정, 프리딜레이 값 설정, 템포 기반 공간계 이펙트 설정에 관한 질문일 때 이 도구를 사용하세요.
    """
    if bpm <= 0:
        return "BPM은 0보다 큰 양수여야 합니다."

    quarter_note = 60000.0 / bpm
    half_note = quarter_note * 2
    eighth_note = quarter_note / 2
    sixteenth_note = quarter_note / 4
    thirty_second_note = quarter_note / 8

    # 프리딜레이 추천 계산 (1/16 음표 기준 1/4 및 1/2 값)
    recommended_pre_delay_short = sixteenth_note / 4
    recommended_pre_delay_long = sixteenth_note / 2

    result = (
        f"입력하신 {bpm} BPM 기준 계산된 공간계 세팅 값입니다:\n"
        f"- 1/2 박자 (이분음표) 딜레이: {half_note:.2f} ms\n"
        f"- 1/4 박자 (사분음표) 딜레이: {quarter_note:.2f} ms\n"
        f"- 1/8 박자 (팔분음표) 딜레이: {eighth_note:.2f} ms\n"
        f"- 1/16 박자 (십육분음표) 딜레이: {sixteenth_note:.2f} ms\n"
        f"- 1/32 박자 (삼십이분음표) 딜레이: {thirty_second_note:.2f} ms\n"
        f"- 추천 보컬 리버브 프리딜레이(Pre-delay): {recommended_pre_delay_short:.2f} ms (빠른 템포/짧은 공간) ~ {recommended_pre_delay_long:.2f} ms (느린 템포/깊은 공간)"
    )
    return result


tools = [search_audio_knowledge, calculate_delay_time]
tool_node = ToolNode(tools)


def analyze_query_node(state: AgentState) -> dict:
    Config.validate()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", google_api_key=Config.GEMINI_API_KEY, temperature=0.1
    )
    parser = JsonOutputParser(pydantic_object=QueryAnalysis)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "사용자의 음향/믹싱 질문을 분석하여 JSON 포맷으로 답해 주세요.\n"
                    "사용자가 구체적인 맥락 (어떤 악기인지, 어떤 이펙터를 쓰고 있는지, 구체적으로 소리가 어떻게 나쁜지 등)을 적지 않고\n"
                    "단순히 '소리가 안 좋아요', '보컬이 묻혀요'와 같이 모호하게 물어보았다면 clarification_needed를 True로 지정하세요.\n\n"
                    "{format_instructions}"
                ),
            ),
            ("human", "{query}"),
        ]
    )

    chain = prompt | llm | parser
    analysis = chain.invoke(
        {
            "query": state["query"],
            "format_instructions": parser.get_format_instructions(),
        }
    )

    return {
        "category": analysis["category"],
        "clarification_needed": analysis["clarification_needed"],
        "collected_details": {
            "instrument": analysis["instrument"],
            "problem_type": analysis["problem_type"],
        },
        "steps": state.get("steps", []) + ["analyze_query"],
    }


def ask_clarification_node(state: AgentState) -> dict:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", google_api_key=Config.GEMINI_API_KEY, temperature=0.5
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "당신은 사운드 믹싱 엔지니어입니다. 사용자가 믹싱에 어려움을 겪고 있으나 정보가 부족합니다.\n"
                    "사용자에게 친절하게 역으로 질문을 던져 구체적인 상황을 파악해야 합니다.\n"
                    "예를 들어 악기의 종류, 사용 중인 플러그인(EQ, 컴프 등), 오디오 소스의 대략적인 상태 등을 물어보세요.\n"
                    "질문은 다정하고 명확하게 한국어로 작성해 주세요."
                ),
            ),
            ("human", "{query}"),
        ]
    )

    response = llm.invoke(prompt.format(query=state["query"]))

    # 역질문의 경우 최종 답변으로 사용할 수 있도록 함
    return {
        "response": response.content,
        "steps": state.get("steps", []) + ["ask_clarification"],
        "messages": [response],
    }


def call_model_node(state: AgentState) -> dict:
    Config.validate()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", google_api_key=Config.GEMINI_API_KEY, temperature=0.3
    )
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = (
        "당신은 전문 사운드 엔지니어입니다. 사용자의 음악 믹싱/마스터링 관련 질문에 답해 주세요.\n"
        "제공된 도구(search_audio_knowledge, calculate_delay_time)를 활용하여 사용자 질문에 정확히 답변해야 합니다.\n"
        "이전 대화 맥락과 제공된 도구의 실행 결과를 적극 활용해 최상의 답변을 한국어로 만들어 주세요."
    )

    current_messages = list(state.get("messages", []))
    messages_to_add = []

    # 툴 호출 턴인지 판별 (이전 메시지의 마지막이 ToolMessage이거나 tool_calls가 있는 AIMessage인 경우)
    is_tool_turn = False
    if current_messages:
        last_msg = current_messages[-1]
        if last_msg.type == "tool" or (
            last_msg.type == "ai"
            and hasattr(last_msg, "tool_calls")
            and last_msg.tool_calls
        ):
            is_tool_turn = True

    # 툴 실행 턴이 아닌 경우(즉 새 쿼리 턴)에만 HumanMessage를 추가
    if not is_tool_turn:
        user_message = HumanMessage(content=state["query"])
        current_messages.append(user_message)
        messages_to_add.append(user_message)

    input_messages = [SystemMessage(content=system_prompt)] + current_messages
    response = llm_with_tools.invoke(input_messages)

    messages_to_add.append(response)

    # response.content가 복합 객체(리스트) 형태일 경우 텍스트 타입 콘텐츠들만 조인하여 문자열로 변환
    content = response.content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        final_response = "".join(text_parts)
    else:
        final_response = str(content) if content else ""

    return {
        "response": "" if response.tool_calls else final_response,
        "messages": messages_to_add,
        "steps": state.get("steps", []) + ["call_model"],
    }


def route_after_analysis(
    state: AgentState,
) -> Literal["ask_clarification", "call_model"]:
    if state["clarification_needed"]:
        return "ask_clarification"
    return "call_model"


def build_agent_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("ask_clarification", ask_clarification_node)
    workflow.add_node("call_model", call_model_node)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("analyze_query")

    workflow.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {"ask_clarification": "ask_clarification", "call_model": "call_model"},
    )

    workflow.add_edge("ask_clarification", END)

    workflow.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    workflow.add_edge("tools", "call_model")
    workflow.add_edge("call_model", END)

    # 메모리 체크포인터 지정을 통해 멀티턴 기억 유지 기능 탑재
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
