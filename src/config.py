import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../chroma_db"))
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

    @classmethod
    def validate(cls):
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY가 존재하지 않습니다. .env 파일을 확인해 주세요."
            )
