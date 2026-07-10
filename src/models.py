from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ChatLog(Base):
    __tablename__ = "chat_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(String(128), index=True)
    query = Column(Text)
    response = Column(Text)
    category = Column(String(32))
    created_at = Column(DateTime, default=datetime.utcnow)


class DocumentMeta(Base):
    __tablename__ = "document_meta"
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), unique=True)
    source_url = Column(String(512))
    category = Column(String(64))
    object_storage_key = Column(String(512))
    collected_at = Column(DateTime, default=datetime.utcnow)
