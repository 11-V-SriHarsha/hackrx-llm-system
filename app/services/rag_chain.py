from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import re
import logging

logger = logging.getLogger(__name__)

def get_rag_chain(vectorstore):
    """Create optimized RAG chain with Llama 4 Scout for fast, concise, accurate answers."""

    # OPTIMAL MODEL SELECTION - Llama 4 Scout for best accuracy
    try:
        llm = ChatGroq(
            temperature=0,
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=350,  # 🔁 was 200
            timeout=45,
            max_retries=2,
            request_timeout=45
        )

        logger.info("🚀 Llama 4 Scout loaded (optimized for concise responses)")
    except Exception as primary_error:
        logger.warning(f"Llama 4 Scout failed: {primary_error}")
        try:
            llm = ChatGroq(
                temperature=0,
                model_name="llama-3.3-70b-versatile",
                api_key=os.getenv("GROQ_API_KEY"),
                max_tokens=200,
                timeout=45,
                max_retries=2,
                request_timeout=45
            )
            logger.info("🔄 Using Llama 3.3 70B as fallback")
        except Exception as fallback_error:
            logger.warning(f"Llama 3.3 70B failed: {fallback_error}")
            llm = ChatGroq(
                temperature=0,
                model_name="llama-3.1-8b-instant",
                api_key=os.getenv("GROQ_API_KEY"),
                max_tokens=200,
                timeout=45,
                max_retries=2,
                request_timeout=45
            )
            logger.info("⚠️ Using Llama 3.1 8B as final fallback")

    # TUNED RETRIEVER FOR SPEED + RELEVANCE
    try:
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,              # 🔁 was 6
                "fetch_k": 12,
                "lambda_mult": 0.75
            }
        )

        logger.info("🔍 Faster MMR retriever configured (k=8)")
    except Exception as e:
        logger.warning(f"MMR failed, using similarity: {e}")
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )
        logger.info("🔍 Similarity retriever fallback (k=6)")

    # NEW PROMPT - ENFORCE CONCISE ANSWERS
    prompt_template = """You are an insurance document analyzer. Extract ONLY the specific information from the policy document.

CRITICAL INSTRUCTIONS:
1. Answer using ONLY the content in the document sections
2. Limit answer to 1–2 sentences
3. Use exact terms, numbers, durations
4. Say "This information is not specified in the policy document" if not found
5. Do NOT explain, speculate, or generalize
6. Be factual, objective, concise

Document Sections:
{context}

Question: {question}

Answer (1–2 sentences only):"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        if not docs:
            return "No relevant policy sections found."

        formatted_sections = []
        seen_content = set()

        try:
            sorted_docs = sorted(docs, key=lambda x: (
                -x.metadata.get('score', 0),
                x.metadata.get('page_number', 999)
            ))
        except:
            sorted_docs = docs

        for i, doc in enumerate(sorted_docs[:4]):
            content = doc.page_content.strip()
            if len(content) < 40 or content in seen_content:
                continue
            seen_content.add(content)

            content = re.sub(r'\s+', ' ', content)
            content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)
            content = re.sub(r'(\d+)\s*%', r'\1%', content)
            content = re.sub(r'Rs\.?\s*(\d+)', r'Rs. \1', content)

            page_info = f"(Page {doc.metadata.get('page_number', 'N/A')})"
            categories = doc.metadata.get('categories', [])
            category_info = f"[{', '.join(categories)}]" if categories != ['general'] else ""
            section_header = f"Section {i+1} {page_info} {category_info}: "
            formatted_sections.append(section_header + content)

            if len(formatted_sections) >= 4:
                break

        return "\n\n".join(formatted_sections) if formatted_sections else "No relevant policy information found."

    def clean_and_validate_answer(answer):
        if not answer or len(answer.strip()) < 10:
            return "This information is not specified in the policy document."

        answer = re.sub(r'\s+', ' ', answer.strip())
        answer = re.sub(r'(\d+)\s*%', r'\1%', answer)
        answer = re.sub(r'Rs\.?\s*(\d+)', r'Rs. \1', answer)
        if not answer.endswith(('.', '!', '?')):
            answer += '.'
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        answer = re.sub(r'(?:Please note|It\'s important to note|According to the document).*?(?=\.|$)', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\s+', ' ', answer).strip()

        # Final truncate safety
        if len(answer) > 320:
            answer = answer[:300].rsplit('. ', 1)[0] + '.'

        return answer

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        | clean_and_validate_answer
    )

    logger.info("⚡ Concise RAG chain ready (under 45s)")
    return rag_chain
