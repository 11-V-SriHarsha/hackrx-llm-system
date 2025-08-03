import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List
import tempfile
import re
import logging

logger = logging.getLogger(__name__)

def process_document_from_url(url: str, timeout: int = 60) -> List:
    """Enhanced document processing optimized for insurance documents."""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_pdf_path = temp_file.name
    
    try:
        logger.info(f"📄 Downloading document from: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        with open(temp_pdf_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=16384):
                if chunk:
                    f.write(chunk)
        
        logger.info("✅ Download successful")
        
        # Load PDF
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        
        if not documents:
            raise ValueError("No content could be extracted from the PDF")
        
        logger.info(f"📖 Extracted {len(documents)} pages from PDF")
        
        # OPTIMIZED CHUNKING for Insurance Documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,      # Optimal size for insurance policies
            chunk_overlap=150,   # Good context overlap
            length_function=len,
            separators=[
                "\n\n\n",  # Section breaks
                "\n\n",    # Paragraph breaks  
                "\n",      # Line breaks
                ". ",      # Sentence endings
                "? ",      # Questions
                "! ",      # Exclamations
                "; ",      # Semicolons
                ", ",      # Commas
                " ",       # Spaces
                ""         # Character fallback
            ]
        )
        
        chunked_docs = text_splitter.split_documents(documents)
        
        # ENHANCED PROCESSING for Insurance Content
        enhanced_chunks = []
        for i, doc in enumerate(chunked_docs):
            content = doc.page_content.strip()
            
            # Skip very short chunks
            if len(content) < 30:
                continue
            
            # Clean content for insurance documents
            content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
            content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)  # Fix word concatenation
            content = re.sub(r'(\d+)\s*%', r'\1%', content)  # Fix percentages
            content = re.sub(r'(\d+)\s*(years?|months?|days?)', r'\1 \2', content)  # Fix periods
            content = re.sub(r'Rs\.?\s*(\d+)', r'Rs. \1', content)  # Fix currency
            
            content = content.strip()
            doc.page_content = content
            
            # Add comprehensive metadata
            doc.metadata.update({
                'chunk_id': i,
                'source': url,
                'chunk_length': len(content),
                'page_number': doc.metadata.get('page', 0)
            })
            
            # SMART CATEGORIZATION for Better Retrieval
            content_lower = content.lower()
            categories = []
            
            # Multi-category assignment for insurance terms
            if any(keyword in content_lower for keyword in ['waiting period', 'wait', 'waiting']):
                categories.append('waiting_period')
            if any(keyword in content_lower for keyword in ['pre-existing', 'ped', 'pre existing']):
                categories.append('pre_existing')
            if any(keyword in content_lower for keyword in ['maternity', 'pregnancy', 'childbirth', 'delivery']):
                categories.append('maternity')
            if any(keyword in content_lower for keyword in ['ayush', 'ayurveda', 'homeopathy', 'unani', 'siddha']):
                categories.append('ayush')
            if any(keyword in content_lower for keyword in ['room rent', 'icu', 'hospital charges']):
                categories.append('room_rent')
            if any(keyword in content_lower for keyword in ['ambulance', 'transport']):
                categories.append('ambulance')
            if any(keyword in content_lower for keyword in ['no claim bonus', 'ncb', 'no claim discount', 'ncd']):
                categories.append('no_claim_bonus')
            if any(keyword in content_lower for keyword in ['organ donor', 'transplant', 'donation']):
                categories.append('organ_donor')
            if any(keyword in content_lower for keyword in ['health check', 'checkup', 'preventive']):
                categories.append('health_checkup')
            if any(keyword in content_lower for keyword in ['portability', 'switch', 'transfer']):
                categories.append('portability')
            if any(keyword in content_lower for keyword in ['joint replacement', 'hernia', 'cataract', 'surgery']):
                categories.append('surgical_procedures')
            if any(keyword in content_lower for keyword in ['grace period', 'grace']):
                categories.append('grace_period')
            
            doc.metadata['categories'] = categories if categories else ['general']
            enhanced_chunks.append(doc)
        
        logger.info(f"✅ Created {len(enhanced_chunks)} optimized chunks")
        return enhanced_chunks
    
    except Exception as e:
        logger.error(f"❌ Error processing document: {str(e)}")
        raise Exception(f"Error processing document: {str(e)}")
    
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)