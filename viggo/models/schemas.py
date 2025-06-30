# viggo/models/schemas.py
from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
    page_number: int

class QueryResponse(BaseModel):
    question: str
    answer: str
    source_pages: list[int]

class DocumentInfoResponse(BaseModel):
    filename: str
    total_pages: int
    content_start_page: int
    content_end_page: int
