from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class DocumentType(str, Enum):
    URL = "url"
    PDF = "pdf"
    QA = "qa"
    TEXT = "text"

class TrainingRequest(BaseModel):
    document_type: DocumentType
    content: str  # URL for websites, base64 for PDFs, text content for Q&A, or raw text for TEXT
    metadata: Optional[dict] = {}

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    max_tokens: Optional[int] = 4000
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    sources: List[dict] = []

class TrainingResponse(BaseModel):
    success: bool
    message: str
    document_id: Optional[str] = None
    chunks_processed: Optional[int] = 0

class HealthResponse(BaseModel):
    status: str
    message: str