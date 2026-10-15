"""
Shared response schemas. Every retrieval tool returns a `ToolResult`
so the LLM always gets data + citation metadata in a consistent shape,
regardless of which backend served the request.
"""

from typing import Any, Literal
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Traceable pointer back to the source of a piece of retrieved data."""

    source_type: Literal["sql", "vector", "graph"]
    source_id: str = Field(..., description="Table+PK, document/chunk id, or node/edge id")
    document_name: str | None = Field(None, description="Original file name, if applicable")
    page_number: int | None = None
    section: str | None = None
    confidence: float | None = Field(None, description="0-1 relevance/similarity score, if applicable")


class ToolError(BaseModel):
    ok: Literal[False] = False
    error: str
    detail: str | None = None


class ToolSuccess(BaseModel):
    ok: Literal[True] = True
    tool: str
    data: list[dict[str, Any]]
    citations: list[Citation]
    row_count: int
    truncated: bool = False


ToolResult = ToolSuccess | ToolError
