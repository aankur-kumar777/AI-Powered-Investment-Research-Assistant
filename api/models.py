# api/models.py
from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

class IngestResponse(BaseModel):
    status: str
    ingested_chunks: int
    collection: str

class HealthResponse(BaseModel):
    status: str = "ok"
