import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    DB_DIR = os.getenv("DB_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "../chroma_db")))
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

    # OCI Object Storage
    OCI_NAMESPACE = os.getenv("OCI_NAMESPACE")
    OCI_BUCKET_NAME = os.getenv("OCI_BUCKET_NAME", "mixmaster-datalake")
    OCI_REGION = os.getenv("OCI_REGION")
    OCI_CONFIG_FILE = os.path.expanduser(os.getenv("OCI_CONFIG_FILE", "~/.oci/config"))

    # MySQL
    MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
    MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
    MYSQL_USER = os.getenv("MYSQL_USER", "mixmaster")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
    MYSQL_DB = os.getenv("MYSQL_DB", "mixmaster")

    @classmethod
    def mysql_url(cls):
        return f"mysql+pymysql://{cls.MYSQL_USER}:{cls.MYSQL_PASSWORD}@{cls.MYSQL_HOST}:{cls.MYSQL_PORT}/{cls.MYSQL_DB}"

    @classmethod
    def validate(cls):
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY가 존재하지 않습니다. .env 파일을 확인해 주세요."
            )
