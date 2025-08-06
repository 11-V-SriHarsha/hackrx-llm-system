from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from app.database import QuerySession, QueryHistory, DocumentMetadata, APIUsage
from typing import List, Dict, Optional
import hashlib
import uuid
import logging

logger = logging.getLogger(__name__)

class DatabaseService:
    
    @staticmethod
    async def create_query_session(
        db: AsyncSession,
        document_url: str,
        document_hash: str,
        pinecone_namespace: str,
        total_questions: int
    ) -> str:
        """Create a new query session and return session ID."""
        session_id = str(uuid.uuid4())
        
        session = QuerySession(
            session_id=session_id,
            document_url=document_url,
            document_hash=document_hash,
            pinecone_namespace=pinecone_namespace,
            total_questions=total_questions,
            processing_time=0.0,  # Will be updated later
            status="processing"
        )
        
        db.add(session)
        await db.commit()
        logger.info(f"📝 Created session: {session_id}")
        return session_id
    
    @staticmethod
    async def update_session_completion(
        db: AsyncSession,
        session_id: str,
        processing_time: float,
        status: str = "completed"
    ):
        """Update session with completion details."""
        result = await db.execute(
            select(QuerySession).where(QuerySession.session_id == session_id)
        )
        session = result.scalar_one_or_none()
        
        if session:
            session.processing_time = processing_time
            session.status = status
            await db.commit()
            logger.info(f"✅ Updated session {session_id}: {processing_time:.2f}s")
    
    @staticmethod
    async def log_query_history(
        db: AsyncSession,
        session_id: str,
        question: str,
        answer: str,
        question_number: int,
        processing_time: float,
        retry_count: int = 0
    ):
        """Log individual question-answer pair."""
        query_log = QueryHistory(
            session_id=session_id,
            question=question,
            answer=answer,
            question_number=question_number,
            processing_time=processing_time,
            retry_count=retry_count
        )
        
        db.add(query_log)
        await db.commit()
        logger.debug(f"💾 Logged Q{question_number} for session {session_id}")
    
    @staticmethod
    async def store_document_metadata(
        db: AsyncSession,
        document_url: str,
        document_hash: str,
        pinecone_namespace: str,
        total_pages: int,
        total_chunks: int,
        chunk_categories: List[str],
        processing_time: float,
        file_size: Optional[int] = None
    ):
        """Store or update document metadata."""
        
        # Check if document already exists
        result = await db.execute(
            select(DocumentMetadata).where(DocumentMetadata.document_hash == document_hash)
        )
        existing_doc = result.scalar_one_or_none()
        
        if existing_doc:
            # Update existing document
            existing_doc.last_accessed = func.now()
            existing_doc.access_count += 1
            await db.commit()
            logger.info(f"📊 Updated document access: {document_url}")
            return existing_doc.id
        else:
            # Create new document record
            doc_metadata = DocumentMetadata(
                document_url=document_url,
                document_hash=document_hash,
                pinecone_namespace=pinecone_namespace,
                total_pages=total_pages,
                total_chunks=total_chunks,
                chunk_categories=chunk_categories,
                file_size=file_size,
                processing_time=processing_time
            )
            
            db.add(doc_metadata)
            await db.commit()
            logger.info(f"📄 Stored document metadata: {document_url}")
            return doc_metadata.id
    
    @staticmethod
    async def log_api_usage(
        db: AsyncSession,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        """Log API usage statistics."""
        usage_log = APIUsage(
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time=response_time,
            user_agent=user_agent,
            ip_address=ip_address,
            error_message=error_message
        )
        
        db.add(usage_log)
        await db.commit()
    
    @staticmethod
    async def get_usage_statistics(db: AsyncSession) -> Dict:
        """Get comprehensive usage statistics."""
        
        # Total sessions
        total_sessions_result = await db.execute(select(func.count(QuerySession.id)))
        total_sessions = total_sessions_result.scalar()
        
        # Total questions
        total_questions_result = await db.execute(select(func.count(QueryHistory.id)))
        total_questions = total_questions_result.scalar()
        
        # Average processing time
        avg_time_result = await db.execute(select(func.avg(QuerySession.processing_time)))
        avg_processing_time = avg_time_result.scalar() or 0
        
        # Most accessed documents
        most_accessed_result = await db.execute(
            select(DocumentMetadata.document_url, DocumentMetadata.access_count)
            .order_by(desc(DocumentMetadata.access_count))
            .limit(5)
        )
        most_accessed_docs = [
            {"url": url, "access_count": count} 
            for url, count in most_accessed_result.fetchall()
        ]
        
        # Recent sessions
        recent_sessions_result = await db.execute(
            select(QuerySession.session_id, QuerySession.document_url, 
                   QuerySession.total_questions, QuerySession.created_at)
            .order_by(desc(QuerySession.created_at))
            .limit(10)
        )
        recent_activity = [
            {
                "session_id": session_id,
                "document_url": url,
                "total_questions": questions,
                "created_at": created_at
            }
            for session_id, url, questions, created_at in recent_sessions_result.fetchall()
        ]
        
        return {
            "total_sessions": total_sessions,
            "total_questions": total_questions,
            "avg_processing_time": round(avg_processing_time, 2),
            "most_accessed_documents": most_accessed_docs,
            "recent_activity": recent_activity
        }
    
    @staticmethod
    async def check_document_cached(db: AsyncSession, document_url: str) -> Optional[Dict]:
        """Check if document is already processed and cached."""
        document_hash = hashlib.md5(document_url.encode()).hexdigest()
        
        result = await db.execute(
            select(DocumentMetadata).where(DocumentMetadata.document_hash == document_hash)
        )
        doc_metadata = result.scalar_one_or_none()
        
        if doc_metadata:
            return {
                "cached": True,
                "namespace": doc_metadata.pinecone_namespace,
                "total_chunks": doc_metadata.total_chunks,
                "last_accessed": doc_metadata.last_accessed
            }
        
        return {"cached": False}