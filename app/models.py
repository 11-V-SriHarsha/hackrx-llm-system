from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL of the document to be processed.")
    questions: List[str] = Field(..., description="List of questions to be answered.")

class QueryResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to the questions.")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds")
    document_cached: Optional[bool] = Field(None, description="Whether document was cached")

# Analytics Models
class SessionStats(BaseModel):
    session_id: str
    document_url: str
    total_questions: int
    processing_time: float
    created_at: datetime
    status: str

class UsageStats(BaseModel):
    total_sessions: int
    total_questions: int
    avg_processing_time: float
    most_accessed_documents: List[dict]
    recent_activity: List[dict]