import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from typing import List, Tuple
import hashlib

def get_vectorstore(chunked_docs: List, document_url: str) -> Tuple[PineconeVectorStore, str]:
    """Create vectorstore using a single persistent index and namespace based on document."""

    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        print("Connected to Pinecone successfully")
    except Exception as e:
        raise Exception(f"Failed to connect to Pinecone: {e}")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 16
            }
        )
        print("Embeddings model loaded successfully")
    except Exception as e:
        raise Exception(f"Failed to load embeddings model: {e}")

    # Use fixed index name and generate namespace per document
    index_name = "hackrx-final"
    namespace = hashlib.md5(document_url.encode()).hexdigest()[:8]
    model_dimension = 384

    try:
        existing_indexes = pc.list_indexes().names()

        if index_name not in existing_indexes:
            try:
                print(f"Creating persistent index: {index_name}")
                pc.create_index(
                    name=index_name,
                    dimension=model_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                time.sleep(10)  # Allow time for index to initialize
            except Exception as create_err:
                raise Exception(f"Unable to create persistent index '{index_name}': {create_err}")

        vectorstore = PineconeVectorStore.from_documents(
            documents=chunked_docs,
            embedding=embeddings,
            index_name=index_name,
            namespace=namespace
        )

        print("Vectorstore created using namespace:", namespace)
        return vectorstore, namespace

    except Exception as e:
        raise Exception(f"Failed to create vectorstore: {e}")

def cleanup_old_indexes():
    """Clean up old random indexes (call this periodically)"""
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        indexes = pc.list_indexes().names()

        cleaned_count = 0
        for index_name in indexes:
            if index_name.startswith("hackrx-rag-"):  # Old random indexes
                try:
                    print(f"Cleaning up old index: {index_name}")
                    pc.delete_index(index_name)
                    cleaned_count += 1
                except Exception as e:
                    print(f"Error deleting index {index_name}: {e}")

        if cleaned_count > 0:
            print(f"Cleaned up {cleaned_count} old indexes")
        else:
            print("No old indexes to clean up")

    except Exception as e:
        print(f"Error during cleanup: {e}")

def delete_pinecone_index(index_name: str):
    """Delete specific Pinecone index with error handling."""
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
