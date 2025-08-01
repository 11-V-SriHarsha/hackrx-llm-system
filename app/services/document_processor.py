import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List
import tempfile

def process_document_from_url(url: str, timeout: int = 60) -> List:
    """Process document from URL with improved error handling and temporary file management."""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_pdf_path = temp_file.name
    
    try:
        print(f"Downloading document from: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Validate content type
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
            print(f"Warning: Content type is {content_type}, proceeding anyway")
        
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
        
        # Improved text splitting with better parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        chunked_docs = text_splitter.split_documents(documents)
        print(f"Created {len(chunked_docs)} text chunks")
        
        return chunked_docs
    
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