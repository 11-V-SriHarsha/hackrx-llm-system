import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from typing import List

def get_vectorstore(chunked_docs: List, index_name: str):
    """Create vectorstore with improved error handling and index readiness check."""
    
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        print("Connected to Pinecone successfully")
    except Exception as e:
        raise Exception(f"Failed to connect to Pinecone: {e}")
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embeddings model loaded successfully")
    except Exception as e:
        raise Exception(f"Failed to load embeddings model: {e}")
    
    model_dimension = 384
    
    try:
        existing_indexes = pc.list_indexes().names()
        
        if index_name not in existing_indexes:
            print(f"Creating new Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=model_dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            
            # Wait for index to be ready with timeout
            max_wait_time = 120  # 2 minutes max
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                try:
                    index_desc = pc.describe_index(index_name)
                    if index_desc.status['ready']:
                        print(f"Index {index_name} is ready!")
                        break
                except Exception:
                    pass
                print(f"Waiting for index {index_name} to be ready...")
                time.sleep(5)
            else:
                raise Exception(f"Index {index_name} did not become ready within {max_wait_time} seconds")
        else:
            print(f"Using existing index: {index_name}")
            
    except Exception as e:
        raise Exception(f"Failed to create/access Pinecone index: {e}")
            
    try:
        print(f"Creating vectorstore with {len(chunked_docs)} documents...")
        vectorstore = PineconeVectorStore.from_documents(
            documents=chunked_docs, 
            embedding=embeddings, 
            index_name=index_name
        )
        print("Vectorstore created successfully")
        return vectorstore
        
    except Exception as e:
        # Clean up index if vectorstore creation fails
        try:
            if index_name in pc.list_indexes().names():
                pc.delete_index(index_name)
                print(f"Cleaned up failed index: {index_name}")
        except:
            pass
        raise Exception(f"Failed to create vectorstore: {e}")

def delete_pinecone_index(index_name: str):
    """Delete Pinecone index with error handling."""
    try:
        print(f"Attempting to delete index: {index_name}")
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        if index_name in pc.list_indexes().names():
            pc.delete_index(index_name)
            print(f"Successfully deleted index: {index_name}")
        else:
            print(f"Index {index_name} not found for deletion")
            
    except Exception as e:
        print(f"Error deleting index {index_name}: {e}")