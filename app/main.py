from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from .models import QueryRequest, QueryResponse
from .auth import verify_token
from .services.document_processor import process_document_from_url
from .services.vector_store_manager import get_vectorstore, delete_pinecone_index
from .services.rag_chain import get_rag_chain
from dotenv import load_dotenv
import uuid
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="HackRx 6.0 - Intelligent Query Retrieval System",
    description="An API to answer questions about documents using a RAG pipeline with Llama-4 Scout.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/hackrx/run", 
          response_model=QueryResponse,
          tags=["Query System"],
          summary="Process a document and answer questions")
async def run_submission(
    request: QueryRequest, 
    background_tasks: BackgroundTasks, 
    token: str = Depends(verify_token)
):
    start_time = time.time()
    # Use shorter UUID for index name to avoid potential issues
    index_name = f"hackrx-rag-{uuid.uuid4().hex[:8]}"
    
    logger.info(f"Starting request processing with index: {index_name}")
    
    try:
        # Input validation
        if not request.documents or not request.questions:
            raise HTTPException(
                status_code=400, 
                detail="Both documents URL and questions are required"
            )
        
        if len(request.questions) == 0:
            raise HTTPException(
                status_code=400, 
                detail="At least one question is required"
            )
            
        logger.info(f"Step 1/5: Processing document for index '{index_name}'...")
        chunked_docs = process_document_from_url(request.documents)
        
        if not chunked_docs:
            raise HTTPException(
                status_code=400, 
                detail="Document is empty or could not be processed."
            )

        logger.info(f"Step 2/5: Creating vector store with {len(chunked_docs)} chunks...")
        vectorstore = get_vectorstore(chunked_docs, index_name)

        logger.info("Step 3/5: Building the RAG chain...")
        rag_chain = get_rag_chain(vectorstore)

        logger.info(f"Step 4/5: Answering {len(request.questions)} questions...")
        answers = []
        
        for i, question in enumerate(request.questions, 1):
            try:
                logger.info(f"Processing question {i}/{len(request.questions)}: {question[:50]}...")
                answer = rag_chain.invoke(question.strip())
                answers.append(answer.strip() if answer else "Unable to generate answer.")
            except Exception as e:
                error_msg = f"Error processing question {i}: {str(e)}"
                logger.error(error_msg)
                answers.append("Sorry, I encountered an error processing this question.")
        
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Step 5/5: Successfully processed request in {processing_time:.2f} seconds.")

        # Schedule cleanup in background
        background_tasks.add_task(delete_pinecone_index, index_name)
        
        return QueryResponse(answers=answers)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        background_tasks.add_task(delete_pinecone_index, index_name)
        raise
        
    except Exception as e:
        error_msg = f"Unexpected error during request processing: {str(e)}"
        logger.error(error_msg, exc_info=True)
        background_tasks.add_task(delete_pinecone_index, index_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=error_msg
        )

@app.get("/", tags=["Health Check"])
async def read_root():
    return {
        "status": "API is running!",
        "message": "HackRx 6.0 - Intelligent Query Retrieval System",
        "endpoints": {
            "main": "/api/v1/hackrx/run",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", tags=["Health Check"])
async def health_check():
    """Comprehensive health check endpoint for debugging."""
    import os
    
    # Check environment variables
    env_checks = {
        "groq_api_key": "configured" if os.getenv("GROQ_API_KEY") else "missing",
        "pinecone_api_key": "configured" if os.getenv("PINECONE_API_KEY") else "missing",
        "bearer_token": "configured" if os.getenv("HACKRX_BEARER_TOKEN") else "missing"
    }
    
    # Test Pinecone connection
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        pc.list_indexes()
        pinecone_status = "connected"
    except Exception as e:
        pinecone_status = f"error: {str(e)[:100]}"
    
    # Test Groq connection
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=os.getenv("GROQ_API_KEY"),
            timeout=10
        )
        # Quick test
        response = llm.invoke("Test connection")
        groq_status = "connected"
    except Exception as e:
        groq_status = f"error: {str(e)[:100]}"
    
    all_healthy = (
        all(status == "configured" for status in env_checks.values()) and
        pinecone_status == "connected" and
        groq_status == "connected"
    )
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": time.time(),
        "environment": env_checks,
        "services": {
            "pinecone": pinecone_status,
            "groq": groq_status
        },
        "system_info": {
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            "fastapi_running": True
        }
    }