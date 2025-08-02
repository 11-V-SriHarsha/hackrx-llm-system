import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List
import tempfile
import re

def process_document_from_url(url: str, timeout: int = 60) -> List:
    """Optimized document processing for better information extraction."""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_pdf_path = temp_file.name
    
    try:
        print(f"Downloading document from: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Write content to temp file
        total_size = 0
        with open(temp_pdf_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        print(f"Downloaded {total_size} bytes successfully")
        
        # Load PDF
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        
        if not documents:
            raise ValueError("No content could be extracted from the PDF")
        
        print(f"Extracted {len(documents)} pages from PDF")
        
        # Optimized text splitting for better fact extraction
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,      # Smaller chunks for precise information
            chunk_overlap=100,   # Sufficient overlap
            length_function=len,
            separators=[
                "\n\n\n",  # Section breaks
                "\n\n",    # Paragraph breaks  
                "\n",      # Line breaks
                ". ",      # Sentence endings
                "? ",      # Question sentences
                "! ",      # Exclamation sentences
                "; ",      # Semicolon breaks
                ": ",      # Colon breaks
                ", ",      # Comma breaks
                " ",       # Space breaks
                ""         # Character level fallback
            ]
        )
        
        chunked_docs = text_splitter.split_documents(documents)
        
        # Enhanced cleaning for better information extraction
        enhanced_chunks = []
        for doc in chunked_docs:
            content = doc.page_content.strip()
            
            # Skip very short chunks
            if len(content) < 40:
                continue
            
            # Advanced cleaning optimized for insurance documents
            content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
            content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)  # Fix concatenated words
            content = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', content)  # Fix number-letter concatenations
            
            # Preserve important insurance terms and numbers
            content = re.sub(r'(\d+)\s*%', r'\1%', content)  # Fix percentage formatting
            content = re.sub(r'(\d+)\s*(years?|days?|months?)', r'\1 \2', content)  # Fix time periods
            content = re.sub(r'(\d+)\s*(lakhs?|crores?)', r'\1 \2', content)  # Fix currency amounts
            
            # Clean up excessive whitespace
            content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
            content = content.strip()
            
            # Update the document content
            doc.page_content = content
            
            # Add metadata for better categorization and retrieval
            doc.metadata.update({
                'chunk_id': len(enhanced_chunks),
                'source': url,
                'chunk_length': len(content),
                'content_type': 'insurance_policy'
            })
            
            # Simple content categorization for better retrieval
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in ['age', 'entry', 'eligibility', 'minimum', 'maximum']):
                doc.metadata['category'] = 'age_eligibility'
            elif any(keyword in content_lower for keyword in ['maturity', 'benefit', 'sum assured', 'death']):
                doc.metadata['category'] = 'benefits'
            elif any(keyword in content_lower for keyword in ['premium', 'payment', 'frequency', 'mode']):
                doc.metadata['category'] = 'premium'
            elif any(keyword in content_lower for keyword in ['rider', 'add-on', 'additional', 'optional']):
                doc.metadata['category'] = 'riders'
            elif any(keyword in content_lower for keyword in ['loan', 'surrender', 'advance']):
                doc.metadata['category'] = 'loan_surrender'
            elif any(keyword in content_lower for keyword in ['free look', 'cancellation', 'cooling']):
                doc.metadata['category'] = 'free_look'
            elif any(keyword in content_lower for keyword in ['suicide', 'exclusion', 'exception']):
                doc.metadata['category'] = 'exclusions'
            elif any(keyword in content_lower for keyword in ['revival', 'lapse', 'reinstatement']):
                doc.metadata['category'] = 'revival'
            elif any(keyword in content_lower for keyword in ['tax', '80c', '10(10d)', 'deduction']):
                doc.metadata['category'] = 'tax_benefits'
            else:
                doc.metadata['category'] = 'general'
            
            enhanced_chunks.append(doc)
        
        print(f"Created {len(enhanced_chunks)} enhanced chunks for better extraction")
        
        return enhanced_chunks
    
    except requests.exceptions.Timeout:
        raise Exception(f"Timeout occurred while downloading document from {url}")
    except requests.exceptions.ConnectionError:
        raise Exception(f"Connection error while downloading document from {url}")
    except requests.exceptions.HTTPError as e:
        raise Exception(f"HTTP error {e.response.status_code} while downloading document")
    except Exception as e:
        raise Exception(f"Error processing document: {str(e)}")
    
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            print("Temporary PDF file cleaned up")