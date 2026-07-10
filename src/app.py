import json
import os
import sys
import time
from typing import Any, AsyncGenerator, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from agent import build_agent_graph
from db import SessionLocal
from models import ChatLog, DocumentMeta
from sqlalchemy import func

app = FastAPI(
    title="MixMaster AI API Server",
    description="사운드 엔지니어링 믹싱 가이드 RAG 에이전트를 위한 백엔드 API",
    version="1.0.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# [Middleware] 로깅, 성능 측정 및 예외 복구를 위한 커스텀 Audit 미들웨어 적용
@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"
        print(
            f"[AUDIT] {request.method} {request.url.path} - 완료 (소요시간: {process_time:.4f}초)"
        )
        return response
    except Exception as e:
        process_time = time.time() - start_time
        print(
            f"[AUDIT ERROR] {request.method} {request.url.path} - 예외 발생: {str(e)} (소요시간: {process_time:.4f}초)"
        )
        return JSONResponse(
            status_code=500,
            content={"detail": f"서버 내부 미들웨어 예외 감지: {str(e)}"},
        )


agent_app = build_agent_graph()

# 간단한 가드레일 검증을 위한 비속어 필터
BAD_WORDS = ["바보", "멍청이", "쓰레기", "욕설"]


def validate_input_guardrail(query: str):
    """입력값에 대한 가드레일 검증 미들웨어 성격의 유효성 검사 함수"""
    for word in BAD_WORDS:
        if word in query:
            raise HTTPException(
                status_code=400,
                detail=f"가드레일 감지: 부적절한 단어('{word}')가 포함되어 있습니다.",
            )


def log_chat(thread_id: str, query: str, response: str, category: str):
    """대화 로그를 MySQL에 적재"""
    db = SessionLocal()
    try:
        db.add(ChatLog(thread_id=thread_id, query=query, response=response, category=category))
        db.commit()
    except Exception as e:
        print(f"[DB LOG ERROR] {e}")
    finally:
        db.close()


class ChatRequest(BaseModel):
    query: str = Field(description="사용자의 사운드 믹싱 질문 텍스트")
    thread_id: str = Field(
        default="default_session",
        description="멀티턴 대화 관리를 위한 고유 세션 ID",
    )


class ChatResponse(BaseModel):
    query: str = Field(description="질문 원본")
    category: str = Field(description="분석된 카테고리")
    clarification_needed: bool = Field(description="역질문 필요 여부")
    collected_details: Dict[str, Any] = Field(description="악기 소스 및 문제 유형 정보")
    response: str = Field(description="최종 답변 본문 (에이전트 조언 또는 역질문)")
    sources: List[str] = Field(description="참조된 RAG 문서 소스 리스트")
    steps: List[str] = Field(description="에이전트가 통과한 내부 LangGraph 노드 로그")


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_chat(query: str, thread_id: str) -> AsyncGenerator[str, None]:
    initial_state = {
        "query": query,
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

    seen_steps: List[str] = []
    final_state: Dict[str, Any] = {}

    config = {"configurable": {"thread_id": thread_id}}

    try:
        async for chunk in agent_app.astream(
            initial_state, config=config, stream_mode="updates"
        ):
            for node_output in chunk.values():
                final_state.update(node_output)
                for step_name in node_output.get("steps", []):
                    if step_name not in seen_steps:
                        seen_steps.append(step_name)
                        yield _sse_event("step", {"steps": list(seen_steps)})

        yield _sse_event(
            "final",
            {
                "query": query,
                "category": final_state.get("category", "general"),
                "clarification_needed": final_state.get("clarification_needed", False),
                "collected_details": final_state.get("collected_details", {}),
                "response": final_state.get("response", ""),
                "sources": final_state.get("sources", []),
                "steps": final_state.get("steps", []),
            },
        )
    except Exception as e:
        yield _sse_event("error", {"detail": f"에이전트 내부 실행 에러: {str(e)}"})


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="질문 내용을 입력해 주세요.")

    # 가드레일 검증
    validate_input_guardrail(request.query)

    try:
        initial_state = {
            "query": request.query,
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

        config = {"configurable": {"thread_id": request.thread_id}}
        final_state = agent_app.invoke(initial_state, config=config)

        response_text = final_state.get("response", "")
        category = final_state.get("category", "general")
        log_chat(request.thread_id, request.query, response_text, category)

        return ChatResponse(
            query=request.query,
            category=category,
            clarification_needed=final_state.get("clarification_needed", False),
            collected_details=final_state.get("collected_details", {}),
            response=response_text,
            sources=final_state.get("sources", []),
            steps=final_state.get("steps", []),
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"에이전트 내부 실행 에러: {str(e)}"
        )


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="질문 내용을 입력해 주세요.")

    # 가드레일 검증
    validate_input_guardrail(request.query)

    return StreamingResponse(
        _stream_chat(request.query, request.thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/stats/chat-summary")
def chat_summary():
    """MySQL에 적재된 세션 로그를 집계하여 반환하는 간단 대시보드 API"""
    db = SessionLocal()
    try:
        total_chats = db.query(func.count(ChatLog.id)).scalar() or 0
        category_stats = db.query(ChatLog.category, func.count(ChatLog.id)).group_by(ChatLog.category).all()
        return {
            "total_chats": total_chats,
            "by_category": {cat: count for cat, count in category_stats}
        }
    except Exception as e:
        print(f"[Stats Error] {e}")
        return {"total_chats": 0, "by_category": {}, "error": str(e)}
    finally:
        db.close()


@app.get("/api/stats/documents")
def document_stats():
    """DocumentMeta 테이블 기반 수집 문서 현황 조회 API"""
    db = SessionLocal()
    try:
        total_docs = db.query(func.count(DocumentMeta.id)).scalar() or 0
        category_stats = db.query(DocumentMeta.category, func.count(DocumentMeta.id)).group_by(DocumentMeta.category).all()
        return {
            "total_documents": total_docs,
            "by_category": {cat: count for cat, count in category_stats}
        }
    except Exception as e:
        print(f"[Stats Error] {e}")
        return {"total_documents": 0, "by_category": {}, "error": str(e)}
    finally:
        db.close()


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "gemini-2.5-flash",
        "embedding": "gemini-embedding-2",
    }


FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend"))
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
