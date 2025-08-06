from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON, Boolean
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=False)

# Create sessionmaker - Python 3.11 compatible syntax
AsyncSessionLocal = sessionmaker(
    bind=engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# Create base class
Base = declarative_base()

# Database Models
class QuerySession(Base):
    __tablename__ = "query_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    document_url = Column(String, nullable=False)
    document_hash = Column(String, nullable=False)
    pinecone_namespace = Column(String, nullable=False)
    total_questions = Column(Integer, nullable=False)
    processing_time = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="completed")

class QueryHistory(Base):
    __tablename__ = "query_history"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    question_number = Column(Integer, nullable=False)
    processing_time = Column(Float, nullable=False)
    retry_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class DocumentMetadata(Base):
    __tablename__ = "document_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    document_url = Column(String, unique=True, index=True)
    document_hash = Column(String, unique=True, index=True)
    pinecone_namespace = Column(String, nullable=False)
    total_pages = Column(Integer, nullable=False)
    total_chunks = Column(Integer, nullable=False)
    chunk_categories = Column(JSON)
    file_size = Column(Integer)
    processing_time = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=1)

class APIUsage(Base):
    __tablename__ = "api_usage"
    
    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String, nullable=False)
    method = Column(String, nullable=False)
    status_code = Column(Integer, nullable=False)
    response_time = Column(Float, nullable=False)
    user_agent = Column(String)
    ip_address = Column(String)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

# Dependency to get database session
async def get_database_session():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Create tables
async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)