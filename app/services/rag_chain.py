from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

def get_rag_chain(vectorstore):
    """Create RAG chain with improved error handling and configuration."""
    
    try:
        llm = ChatGroq(
            temperature=0, 
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=1024,
            timeout=60,
            max_retries=3
        )
        print("Successfully initialized ChatGroq with Llama-4 Scout")
    except Exception as e:
        raise Exception(f"Failed to initialize ChatGroq: {e}")

    try:
        # Enhanced retriever configuration
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}  # Get top 8 most relevant chunks
        )
        print("Successfully created document retriever")
    except Exception as e:
        raise Exception(f"Failed to create retriever: {e}")
    
    # Improved prompt template for better results
    prompt_template = """You are an expert AI assistant specializing in insurance policy analysis.

Your task: Answer the question based SOLELY on the provided context from the insurance document.

Instructions:
1. Use ONLY the information provided in the context below
2. Be precise, accurate, and direct in your answers
3. If the answer is not available in the context, respond exactly: "The answer is not available in the provided document."
4. Do not include document references, chunk IDs, or source mentions in your response
5. Focus specifically on what the question asks
6. Provide complete answers with relevant details when information is available

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    def format_docs(docs):
        """Format retrieved documents for better context presentation."""
        if not docs:
            return "No relevant context found."
        
        # Clean and format the context
        formatted_context = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            if content:
                formatted_context.append(f"Context {i}:\n{content}")
        
        return "\n\n".join(formatted_context) if formatted_context else "No relevant context found."
    
    # Build the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("RAG chain created successfully")
    return rag_chain