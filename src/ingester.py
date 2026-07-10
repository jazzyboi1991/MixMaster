import wikipediaapi
from config import Config
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

TARGET_PAGES = [
    # 음향 공학 및 기초 기술 이론
    "Audio mixing",
    "Mastering (audio)",
    "Equalization (audio)",
    "Dynamic range compression",
    "Reverb",
    "Delay (audio effect)",
    "Phase (waves)",
    "Limiter (audio)",
    "Noise gate",
    "De-essing",
    "Comb filter",
    "Fletcher–Munson curves",
    "Decibel",
    "Stereo imaging",
    "Panning (audio)",
    "Distortion (music)",
    "Chorus effect",
    # 글로벌 주요 플러그인/음향 기기 제조 회사
    "iZotope",
    "Universal Audio (company)",
    "Solid State Logic",
    "Dolby Laboratories",
    "Roland Corporation",
    "Moog Music",
    "Sennheiser",
    "Shure",
    "Lexicon (company)",
    "Focusrite",
    "Native Instruments",
    "Arturia",
    "Eventide (company)",
    "IK Multimedia",
    "Slate Digital",
    "PreSonus",
    "Mackie",
    "Behringer",
    "Yamaha Corporation",
    "Neumann (company)",
    "AKG (company)",
    "Audio-Technica",
    "Beyerdynamic",
    "JBL",
    "Korg",
    "Novation Digital Music Systems",
    "Genelec",
    "KRK Systems",
    "Adam Audio",
    # 산업 표준 소프트웨어 및 명기 하드웨어
    "Auto-Tune",
    "Melodyne",
    "LA-2A Levelling Amplifier",
    "1176 Peak Limiter",
    "Distressor",
    "Pro Tools",
    "Ableton Live",
    "Logic Pro",
    "FL Studio",
    "Cubase",
    "Reaper (software)",
]


def fetch_wikipedia_content(page_title: str) -> str:
    wiki = wikipediaapi.Wikipedia(
        user_agent="AudioMixingRAGAgent/1.0 (contact@example.com)", language="en"
    )
    page = wiki.page(page_title)
    if not page.exists():
        print(f"경고: 페이지를 찾을 수 없습니다: {page_title}")
        return ""
    return page.text


def build_vector_store():
    Config.validate()

    raw_documents = []
    metadata_list = []

    for title in TARGET_PAGES:
        print(f"'{title}' 수집 시작...")
        content = fetch_wikipedia_content(title)
        if content:
            raw_documents.append(content)
            metadata_list.append({"source": f"Wikipedia: {title}"})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.create_documents(
        texts=raw_documents, metadatas=metadata_list
    )
    print(f"생성된 청크 수: {len(chunks)}개")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2", google_api_key=Config.GEMINI_API_KEY
    )

    print("Vector DB 구축 중...")
    Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=Config.DB_DIR
    )
    print(f"성공적으로 DB를 구축하고 '{Config.DB_DIR}에 저장했습니다.")


if __name__ == "__main__":
    build_vector_store()
