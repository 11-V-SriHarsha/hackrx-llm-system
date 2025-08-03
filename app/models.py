from pydantic import BaseModel, Field
from typing import List

class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL of the document to be processed.")
    questions: List[str] = Field(..., description="List of questions to be answered.")

class QueryResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to the questions.")
