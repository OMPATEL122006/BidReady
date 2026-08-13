import os
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from src.config import DB_DIR, EMBEDDING_MODEL_NAME

def get_chroma_client():
    """
    Initializes a persistent ChromaDB client saved on disk in the DB_DIR folder.
    """
    # Ensure the database directory exists
    os.makedirs(DB_DIR, exist_ok=True)
    
    # Create persistent client
    return chromadb.PersistentClient(path=DB_DIR)

def get_or_create_collection(client, collection_name: str = "tender_requirements"):
    """
    Retrieves an existing collection or creates a new one using the local sentence-transformers model.
    """
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    return client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

def store_chunks_in_chroma(collection, chunks: list):
    """
    Adds document chunks to the ChromaDB collection.
    
    Args:
        collection: The Chroma collection object.
        chunks (list): List of dictionaries, each containing 'chunk_id', 'text', and 'metadata'.
    """
    ids = []
    documents = []
    metadatas = []
    
    for c in chunks:
        ids.append(f"chunk_{c['chunk_id']}")
        documents.append(c["text"])
        
        # Flatten metadata and convert all fields to string/int/float (Chroma requirement)
        meta = {
            "page_number": int(c["metadata"]["page_number"]),
            "char_start": int(c["metadata"]["char_start"]),
            "char_end": int(c["metadata"]["char_end"])
        }
        metadatas.append(meta)
        
    # Chroma handles embedding generation automatically under the hood
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

def query_chroma(collection, query_text: str, n_results: int = 3) -> dict:
    """
    Queries the Chroma collection for the semantically closest text chunks.
    
    Returns a dictionary containing:
        - 'documents': list of retrieved chunk texts
        - 'metadatas': list of metadata dicts
        - 'distances': distance scores (lower means more similar)
    """
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results
