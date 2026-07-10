from typing import Any, Dict, List, TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    query: str
    category: str
    clarification_needed: bool
    collected_details: Dict[str, Any]
    context_documents: List[str]
    sources: List[str]
    response: str
    steps: List[str]
    grade_results: List[str]
    rewrite_attempts: int
    messages: Annotated[list, add_messages]
