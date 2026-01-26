from pydantic import BaseModel
from typing import List, Optional


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int
    status: str


class AskRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None


class Source(BaseModel):
    page: int
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    doc_id: str


class DeleteResponse(BaseModel):
    status: str
    message: str


class ListResponse(BaseModel):
    documents: List[dict]
    total: int
