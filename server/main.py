from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from opsfusion_rag.utils.config import load_config
from opsfusion_rag.pipeline.query import retrieve_and_answer

cfg = load_config()
app = FastAPI(title="EasyRAG LangChain (DeepSeek + Qdrant)")


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    answer = retrieve_and_answer(cfg, req.query)
    return QueryResponse(answer=answer)
