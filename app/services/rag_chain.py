from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import re

def get_rag_chain(vectorstore):
    """Create RAG chain optimized for concise, document-specific answers."""
    
    try:
        # Optimized LLM settings for concise responses
        llm = ChatGroq(
            temperature=0.0,  # Zero temperature for consistent, factual responses
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=150,   # Reduced for concise responses
            timeout=30,
            max_retries=2
        )
        print("Successfully initialized ChatGroq with Llama-4 Scout")
    except Exception as e:
        raise Exception(f"Failed to initialize ChatGroq: {e}")

    try:
        # Enhanced retriever for maximum relevant context
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,  # Sufficient relevant chunks
                "lambda_mult": 0.9,  # Strong preference for relevance
                "fetch_k": 20  # Cast wide net for candidates
            }
        )
        print("Successfully created document retriever")
    except Exception as e:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}
        )
        print("Using similarity search retriever as fallback")
    
    # **KEY FIX**: Completely rewritten prompt for concise, document-focused responses
    prompt_template = """You are an insurance document analyzer. Extract ONLY the specific information from the policy document to answer the question.

CRITICAL INSTRUCTIONS:
1. Answer ONLY with information found in the provided document context
2. Be concise - maximum 1-2 sentences
3. Use exact details, numbers, and terms from the document
4. If information is not in the document, say "This information is not specified in the policy document"
5. Do NOT provide general insurance knowledge or explanations
6. Do NOT use phrases like "typically" or "usually" or "standard practice"
7. Focus on factual extraction, not interpretation

Document Context:
{context}

Question: {question}

Concise Answer (extract directly from document):"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    def format_docs(docs):
        """Format documents for focused extraction."""
        if not docs:
            return "No relevant information found in the policy document."
        
        # Get most relevant content
        formatted_sections = []
        seen_content = set()
        
        for doc in docs[:6]:  # Limit to most relevant docs
            content = doc.page_content.strip()
            
            # Skip duplicates and very short content
            if len(content) < 30 or content in seen_content:
                continue
                
            seen_content.add(content)
            
            # Clean content for better extraction
            content = re.sub(r'\s+', ' ', content)
            content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)
            content = content.strip()
            
            formatted_sections.append(content)
        
        if not formatted_sections:
            return "No relevant information found in the policy document."
            
        return "\n\n".join(formatted_sections)
    
    def clean_answer(answer):
        """Clean and ensure concise answer format."""
        if not answer:
            return "This information is not specified in the policy document."
        
        # Remove verbose patterns
        answer = re.sub(r'The (provided )?policy document (does not explicitly mention|does not mention|provided does not)', 
                       'This policy does not specify', answer)
        answer = re.sub(r'typically.*?policies', '', answer)
        answer = re.sub(r'usually.*?insurance', '', answer)
        answer = re.sub(r'standard.*?practice', '', answer)
        answer = re.sub(r'in the insurance industry', '', answer)
        
        # Convert to single line
        answer = re.sub(r'\n+', ' ', answer)
        answer = re.sub(r'\s+', ' ', answer)
        answer = answer.strip()
        
        # Ensure proper ending
        if answer and not answer.endswith('.'):
            answer += '.'
        
        # Capitalize first letter
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        
        # Limit length - if too long, extract first sentence
        if len(answer) > 300:
            sentences = answer.split('. ')
            if len(sentences) > 1:
                answer = sentences[0] + '.'
        
        return answer
    
    # Build the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        | clean_answer
    )
    
    print("Concise document-focused RAG chain created successfully")
    return rag_chain