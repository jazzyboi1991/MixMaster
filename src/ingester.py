__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import wikipediaapi
from datetime import datetime
from config import Config
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from db import SessionLocal
from models import DocumentMeta
from oci_storage import upload_text
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TARGET_PAGES_WITH_CATEGORY = [
    # 음향 공학 및 기초 기술 이론
    ("Audio mixing", "theory"),
    ("Mastering (audio)", "theory"),
    ("Equalization (audio)", "theory"),
    ("Dynamic range compression", "theory"),
    ("Reverb", "theory"),
    ("Delay (audio effect)", "theory"),
    ("Phase (waves)", "theory"),
    ("Limiter (audio)", "theory"),
    ("Noise gate", "theory"),
    ("De-essing", "theory"),
    ("Comb filter", "theory"),
    ("Fletcher–Munson curves", "theory"),
    ("Decibel", "theory"),
    ("Stereo imaging", "theory"),
    ("Panning (audio)", "theory"),
    ("Distortion (music)", "theory"),
    ("Chorus effect", "theory"),
    # 글로벌 주요 플러그인/음향 기기 제조 회사
    ("iZotope", "manufacturer"),
    ("Universal Audio (company)", "manufacturer"),
    ("Solid State Logic", "manufacturer"),
    ("Dolby Laboratories", "manufacturer"),
    ("Roland Corporation", "manufacturer"),
    ("Moog Music", "manufacturer"),
    ("Sennheiser", "manufacturer"),
    ("Shure", "manufacturer"),
    ("Lexicon (company)", "manufacturer"),
    ("Focusrite", "manufacturer"),
    ("Native Instruments", "manufacturer"),
    ("Arturia", "manufacturer"),
    ("Eventide (company)", "manufacturer"),
    ("IK Multimedia", "manufacturer"),
    ("Slate Digital", "manufacturer"),
    ("PreSonus", "manufacturer"),
    ("Mackie", "manufacturer"),
    ("Behringer", "manufacturer"),
    ("Yamaha Corporation", "manufacturer"),
    ("Neumann (company)", "manufacturer"),
    ("AKG (company)", "manufacturer"),
    ("Audio-Technica", "manufacturer"),
    ("Beyerdynamic", "manufacturer"),
    ("JBL", "manufacturer"),
    ("Korg", "manufacturer"),
    ("Novation Digital Music Systems", "manufacturer"),
    ("Genelec", "manufacturer"),
    ("KRK Systems", "manufacturer"),
    ("Adam Audio", "manufacturer"),
    # 산업 표준 소프트웨어 및 명기 하드웨어
    ("Auto-Tune", "software"),
    ("Melodyne", "software"),
    ("LA-2A Levelling Amplifier", "software"),
    ("1176 Peak Limiter", "software"),
    ("Distressor", "software"),
    ("Pro Tools", "software"),
    ("Ableton Live", "software"),
    ("Logic Pro", "software"),
    ("FL Studio", "software"),
    ("Cubase", "software"),
    ("Reaper (software)", "software"),
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
    db = SessionLocal()

    for title, category in TARGET_PAGES_WITH_CATEGORY:
        print(f"'{title}' 수집 시작...")
        content = fetch_wikipedia_content(title)
        if content:
            raw_documents.append(content)
            metadata_list.append({"source": f"Wikipedia: {title}", "category": category})

            # Object Storage에 원문 저장
            os_key = f"raw/wikipedia/{title}.txt"
            upload_text(os_key, content)

            # MySQL에 메타데이터 적재
            try:
                doc_meta = DocumentMeta(
                    title=title,
                    source_url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    category=category,
                    object_storage_key=os_key,
                    collected_at=datetime.utcnow()
                )
                db.add(doc_meta)
            except Exception as e:
                print(f"[DB Error] {title}: {str(e)}")

    try:
        db.commit()
    except Exception as e:
        print(f"[DB Commit Error] {str(e)}")
        db.rollback()
    finally:
        db.close()

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
    print(f"성공적으로 DB를 구축하고 '{Config.DB_DIR}'에 저장했습니다.")


if __name__ == "__main__":
    build_vector_store()
