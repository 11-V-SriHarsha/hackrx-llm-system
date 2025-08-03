import os
import time
import hashlib
import torch
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

def get_vectorstore(chunked_docs: List, document_url: str) -> Tuple[PineconeVectorStore, str]:
    """Create persistent vectorstore with document-specific namespaces."""

    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        logger.info("🔗 Connected to Pinecone successfully")
    except Exception as e:
        raise Exception(f"Failed to connect to Pinecone: {e}")

    try:
        # Correct model: BGE Small (384-d)
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True,  'batch_size': 32}
        )
        logger.info("⚡ Fast embedding model loaded: BAAI/bge-small-en-v1.5 (384-d)")

    except Exception as e:
        raise Exception(f"Failed to load embedding model: {e}")

    # IMPORTANT: Index must match BGE's 384-dim output
    index_name = "hackrx-fast-384"
    namespace = hashlib.md5(document_url.encode()).hexdigest()[:12]

    try:
        existing_indexes = pc.list_indexes().names()

        if index_name not in existing_indexes:
            logger.info(f"🏗️ Creating PERSISTENT index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=384,  # 🔧 FIXED: Was incorrectly set to 768
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info(f"✅ Waiting for index '{index_name}' to be ready...")
            for _ in range(30):  # Wait up to ~30 seconds
                if index_name in pc.list_indexes().names():
                    break
                time.sleep(1)

        vectorstore_index = pc.Index(index_name)

        stats = vectorstore_index.describe_index_stats()
        namespace_stats = stats.get("namespaces", {}).get(namespace, {})
        vector_count = namespace_stats.get("vector_count", 0)

        if vector_count > 0:
            vectorstore = PineconeVectorStore.from_existing_index(
                index_name=index_name,
                embedding=embeddings,
                namespace=namespace
            )
            logger.info(f"♻️ Reusing existing namespace '{namespace}' with {vector_count} vectors")
        else:
            vectorstore = PineconeVectorStore.from_documents(
                documents=chunked_docs,
                embedding=embeddings,
                index_name=index_name,
                namespace=namespace
            )
            logger.info(f"🆕 Created new namespace '{namespace}' with {len(chunked_docs)} vectors")

        return vectorstore, namespace

    except Exception as e:
        raise Exception(f"Failed to create vectorstore: {e}")
