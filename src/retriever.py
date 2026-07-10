from config import Config
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


def ask_audio_agent(query: str):
    Config.validate()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2", google_api_key=Config.GEMINI_API_KEY
    )
    db = Chroma(persist_directory=Config.DB_DIR, embedding_function=embeddings)

    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "당신은 사운드 믹싱 및 마스터링을 가르치는 전문 엔지니어입니다.\n"
                    "주어진 컨텍스트(Context)를 적극적으로 해석하고 요약하여, 사용자의 질문에 전문적인 한국어로 답변해 주세요.\n"
                    "컨텍스트에 언급된 핵심 기술 단어(예: Side-chaining, threshold, ratio 등)의 정의나 설명이 있다면,\n"
                    "이를 바탕으로 원리와 구체적인 활용법을 아주 쉽고 상세하게 설명해 주어야 합니다.\n"
                    "만약 주어진 컨텍스트에서 전혀 관련 정보를 유추할 수 없는 경우에만 답변을 거부해 주세요.\n\n"
                    "Context:\n{context}"
                ),
            ),
            ("human", "{question}"),
        ]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", google_api_key=Config.GEMINI_API_KEY, temperature=0.2
    )

    chain = prompt_template | llm
    response = chain.invoke({"context": context, "question": query})

    print(f"■ 질문: {query}")
    print(f"■ 답변:\n{response.content}\n")
    print(f"■ 참조 문서: {', '.join(sources)}")


if __name__ == "__main__":
    ask_audio_agent("Explain sidechain compression and when should I use it.")
