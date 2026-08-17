import os
import re
import chromadb
from typing import List, Dict, Any, Optional
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from app.config.settings import DB_DIR, EMBEDDING_MODEL_NAME
from app.models.chunk import Chunk
from app.config.logging import logger

class VectorStore:
    """
    ChromaDB wrapper providing persistent vector storage with
    mandatory document_id metadata filtering for strict document isolation.
    """
    def __init__(self, db_dir: str = DB_DIR, collection_name: str = "tender_requirements"):
        self.db_dir = db_dir
        self.collection_name = collection_name
        os.makedirs(self.db_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )

    def store_chunks(self, chunks: List[Chunk]):
        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for c in chunks:
            source_name = c.source_doc or "doc"
            safe_source = re.sub(r'[^a-zA-Z0-9_]', '_', source_name)
            ids.append(f"chunk_{safe_source}_{c.document_id}_{c.chunk_id}")
            documents.append(c.text)
            
            meta = c.to_metadata_dict()
            metadatas.append(meta)

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Stored {len(chunks)} chunks in Chroma collection '{self.collection_name}'.")

    def query(self, query_text: str, n_results: int = 50, document_id: Optional[str] = None, tender_id: Optional[str] = None) -> Dict[str, Any]:
        where_clause = None
        if tender_id and document_id:
            where_clause = {"$and": [{"tender_id": str(tender_id)}, {"document_id": str(document_id)}]}
        elif tender_id:
            where_clause = {"tender_id": str(tender_id)}
        elif document_id:
            where_clause = {"document_id": str(document_id)}

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_clause
        )
        return results

    def clear_database(self):
        try:
            existing = self.collection.get()
            if existing and existing.get("ids"):
                self.collection.delete(ids=existing["ids"])
        except Exception:
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
        logger.info("Chroma database collection cleared.")
