#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from db import engine
from models import Base

if __name__ == "__main__":
    print("MySQL 테이블 생성 중...")
    Base.metadata.create_all(engine)
    print("테이블 생성 완료.")
