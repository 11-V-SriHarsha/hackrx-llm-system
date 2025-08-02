from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from .models import QueryRequest, QueryResponse
from .auth import verify_token
from .services.document_processor import process_document_from_url
from .services.vector_store_manager import get_vectorstore, cleanup_old_indexes
from .services.rag_chain import get_rag_chain
from dotenv import load_dotenv
import time
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="HackRx 6.0 - Concise Document Query System",
    description="API for extracting specific information from insurance documents with concise responses.",
    version="2.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def enhance_question_for_better_retrieval(question: str) -> str:
    """Enhance questions for better document retrieval without adding verbose context."""
    
    # Simple keyword mapping for better retrieval
    enhancements = {
        'entry age': 'minimum maximum entry age eligibility',
        'maturity benefit': 'maturity benefit sum assured policy term',
        'death benefit': 'death benefit sum assured nominee',
        'rider': 'rider add-on additional benefit',
        'premium': 'premium payment frequency terms',
        'policy loan': 'policy loan advance surrender',
        'free look': 'free look period cancellation',
        'suicide': 'suicide exclusion clause',
        'revival': 'revival reinstatement lapse',
        'tax benefit': 'tax benefit 80C 10(10D) deduction'
    }
    
    question_lower = question.lower()
    for key, terms in enhancements.items():
        if key in question_lower:
            return f"{question} {terms}"
    
    return question

def validate_and_clean_answer(answer: str, question: str) -> str:
    """Validate and ensure the answer is concise and document-focused."""
    
    if not answer or len(answer.strip()) < 5:
        return "This information is not specified in the policy document."
    
    # Remove common verbose patterns
    verbose_patterns = [
        r'The provided policy document does not explicitly mention.*?However,',
        r'Typically.*?policies',
        r'Usually.*?insurance',
        r'In the insurance industry.*?',
        r'Standard.*?practice',
        r'Based on.*?practices',
        r'This exclusion suggests.*?',
        r'For detailed.*?document',
        r'It is recommended.*?',
        r'Please refer.*?',
        r'You might be interested.*?',
        r'If you\'re looking for.*?',
    ]
    
    for pattern in verbose_patterns:
        answer = re.sub(pattern, '', answer, flags=re.IGNORECASE | re.DOTALL)
    
    # Clean formatting
    answer = re.sub(r'\s+', ' ', answer)
    answer = answer.strip()
    
    # If answer starts with negative information, try to extract any positive info
    if answer.lower().startswith(('the policy does not', 'this policy does not', 'no mention')):
        # Look for any specific information that might be buried
        positive_patterns = [
            r'however[,\s]+(.*?)(?:\.|$)',
            r'but[,\s]+(.*?)(?:\.|$)',
            r'although[,\s]+(.*?)(?:\.|$)',
        ]
        for pattern in positive_patterns:
            match = re.search(pattern, answer, re.IGNORECASE | re.DOTALL)
            if match:
                positive_info = match.group(1).strip()
                if len(positive_info) > 20:
                    answer = positive_info
                    break
    
    # Ensure proper sentence structure
    if answer and not answer.endswith('.'):
        answer += '.'
    
    # Capitalize first letter
    if answer and answer[0].islower():
        answer = answer[0].upper() + answer[1:]
    
    # Final length check - if still too long, extract first meaningful sentence
    if len(answer) > 250:
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        if sentences:
            answer = sentences[0]
            if not answer.endswith(('.', '!', '?')):
                answer += '.'
    
    return answer

@app.post("/api/v1/hackrx/run", 
          response_model=QueryResponse,
          tags=["Query System"],
          summary="Extract specific information from insurance documents")
async def run_submission(
    request: QueryRequest, 
    background_tasks: BackgroundTasks, 
    token: str = Depends(verify_token)
):
    start_time = time.time()
    
    logger.info(f"Starting document processing for concise extraction...")
    
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
            
        logger.info(f"Step 1/5: Processing document...")
        chunked_docs = process_document_from_url(request.documents)
        
        if not chunked_docs:
            raise HTTPException(
                status_code=400, 
                detail="Document is empty or could not be processed."
            )

        logger.info(f"Step 2/5: Creating vector store with {len(chunked_docs)} chunks...")
        vectorstore, index_name = get_vectorstore(chunked_docs, request.documents)

        logger.info("Step 3/5: Building RAG chain for document extraction...")
        rag_chain = get_rag_chain(vectorstore)

        logger.info(f"Step 4/5: Extracting answers for {len(request.questions)} questions...")
        answers = []
        
        for i, question in enumerate(request.questions, 1):
            try:
                logger.info(f"Processing question {i}/{len(request.questions)}: {question[:50]}...")
                
                # Enhance question for better retrieval
                enhanced_question = enhance_question_for_better_retrieval(question.strip())
                
                # Get answer using enhanced question
                raw_answer = rag_chain.invoke(enhanced_question)
                
                # Validate and clean the answer
                final_answer = validate_and_clean_answer(raw_answer, question)
                
                # If answer is too generic, try with original question
                if final_answer == "This information is not specified in the policy document.":
                    fallback_answer = rag_chain.invoke(question.strip())
                    fallback_cleaned = validate_and_clean_answer(fallback_answer, question)
                    if len(fallback_cleaned) > len(final_answer):
                        final_answer = fallback_cleaned
                
                answers.append(final_answer)
                logger.info(f"Question {i} processed - Answer length: {len(final_answer)} chars")
                
            except Exception as e:
                error_msg = f"Error processing question {i}: {str(e)}"
                logger.error(error_msg)
                answers.append("This information is not specified in the policy document.")
        
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Step 5/5: Completed in {processing_time:.2f} seconds using index: {index_name}")

        # Schedule cleanup in background
        background_tasks.add_task(cleanup_old_indexes)
        
        return QueryResponse(answers=answers)

    except HTTPException:
        raise
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=error_msg
        )

@app.get("/", tags=["Health Check"])
async def read_root():
    return {
        "status": "Concise Document Extraction API is running!",
        "message": "HackRx 6.0 - Optimized for concise, document-specific answers",
        "version": "2.1.0-concise",
        "key_optimizations": [
            "Zero temperature for consistent factual responses",
            "Maximum 150 tokens for concise answers",
            "Document-focused prompt engineering",
            "Verbose pattern removal",
            "Direct information extraction",
            "Fallback for better coverage"
        ],
        "response_format": "Single sentences with direct document information",
        "endpoints": {
            "main": "/api/v1/hackrx/run",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", tags=["Health Check"])
async def health_check():
    """Health check endpoint."""
    import os
    
    # Check environment variables
    env_checks = {
        "groq_api_key": "configured" if os.getenv("GROQ_API_KEY") else "missing",
        "pinecone_api_key": "configured" if os.getenv("PINECONE_API_KEY") else "missing",
        "bearer_token": "configured" if os.getenv("HACKRX_BEARER_TOKEN") else "missing"
    }
    
    all_configured = all(status == "configured" for status in env_checks.values())
    
    return {
        "status": "healthy" if all_configured else "degraded",
        "timestamp": time.time(),
        "environment": env_checks,
        "optimization_settings": {
            "temperature": 0.0,
            "max_tokens": 150,
            "retrieval_strategy": "MMR with k=8",
            "response_style": "concise document extraction",
            "verbose_filtering": "enabled",
            "fallback_mechanism": "enabled"
        }
    }