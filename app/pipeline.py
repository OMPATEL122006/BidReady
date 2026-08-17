import os
import hashlib
from typing import List, Dict, Any, Optional
from app.models.document import DocumentMetadata, TenderDocument, TenderDocumentSet, DocumentType
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.excel_parser import ExcelParser
from app.ingestion.classifier import DocumentClassifier
from app.chunking.text_chunker import TextChunker
from app.storage.vector_store import VectorStore
from app.storage.bm25_store import BM25Store
from app.retrieval.hybrid_search import HybridSearchEngine
from app.generation.answer_generator import AnswerGenerator
from app.config.logging import logger

class TenderRAGPipeline:
    """
    Master Tender RAG Pipeline orchestrating ingestion, classification,
    tender-set boundary isolation, multi-document conflict detection,
    and grounded answer generation.
    """
    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25_store = BM25Store()
        self.text_chunker = TextChunker()
        self.search_engine = HybridSearchEngine(vector_store=self.vector_store, bm25_store=self.bm25_store)
        self.answer_generator = AnswerGenerator(search_engine=self.search_engine)
        self.indexed_chunks = []
        self.tender_sets: Dict[str, TenderDocumentSet] = {}

    def _generate_doc_id(self, file_path: str) -> str:
        fname = os.path.basename(file_path)
        return hashlib.md5(fname.encode("utf-8")).hexdigest()[:12]

    def _generate_tender_id(self, identifier: str) -> str:
        return "tset_" + hashlib.md5(identifier.encode("utf-8")).hexdigest()[:10]

    def ingest_file(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        tender_id: Optional[str] = None,
        override_doc_type: Optional[DocumentType] = None
    ) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        fname = os.path.basename(file_path)
        doc_id = document_id or self._generate_doc_id(file_path)
        t_id = tender_id or "default_tender"
        ext = os.path.splitext(file_path)[1].lower()

        logger.info(f"Ingesting file '{fname}' (tender_id: {t_id}, doc_id: {doc_id})...")

        if ext == ".pdf":
            parser = PDFParser(file_path)
            pages = parser.parse()
        elif ext in [".xlsx", ".xls"]:
            parser = ExcelParser(file_path)
            pages = parser.parse()
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Classify document type based on content sample
        sample_text = "\n".join([el.get("text", "") for p in pages.values() for el in p])[:3000]
        doc_type = override_doc_type or DocumentClassifier.classify(file_path, text_sample=sample_text)

        chunks = self.text_chunker.chunk_document(
            parsed_data=pages,
            source_doc=fname,
            document_id=doc_id,
            tender_id=t_id,
            document_type=doc_type.value if isinstance(doc_type, DocumentType) else str(doc_type)
        )
        self.indexed_chunks.extend(chunks)

        self.vector_store.store_chunks(chunks)
        self.bm25_store.fit(self.indexed_chunks)

        logger.info(f"Successfully ingested '{fname}' [{doc_type.value}]: generated {len(chunks)} chunks.")
        return doc_id

    def ingest_tender_set(self, files_or_dir: List[str] | str, tender_id: Optional[str] = None) -> str:
        """
        Ingests multiple related documents (NIT, Detailed Tender, BOQ, Corrigendum)
        under a single logical TenderDocumentSet with strict boundary isolation.
        """
        if isinstance(files_or_dir, str) and os.path.isdir(files_or_dir):
            target_dir = files_or_dir
            t_id = tender_id or self._generate_tender_id(os.path.basename(target_dir))
            files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.lower().endswith((".pdf", ".xlsx", ".xls"))]
        elif isinstance(files_or_dir, list):
            files = files_or_dir
            t_id = tender_id or self._generate_tender_id("_".join([os.path.basename(f) for f in files[:3]]))
        else:
            raise ValueError("Expected a directory path or list of file paths.")

        logger.info(f"Ingesting Tender Set '{t_id}' ({len(files)} files)...")

        for fp in files:
            self.ingest_file(fp, tender_id=t_id)

        return t_id

    def ingest_directory(self, dir_path: str) -> List[str]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        doc_ids = []
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith((".pdf", ".xlsx", ".xls"))]
        for fp in files:
            d_id = self.ingest_file(fp)
            doc_ids.append(d_id)
        return doc_ids

    def ask(self, query: str, n_results: int = 3, document_id: Optional[str] = None, tender_id: Optional[str] = None) -> Dict[str, Any]:
        return self.answer_generator.generate_answer(query, n_results=n_results, document_id=document_id, tender_id=tender_id)

    def clear(self):
        self.vector_store.clear_database()
        self.indexed_chunks = []
        self.bm25_store.fit([])
        self.tender_sets = {}
        logger.info("Pipeline database and indices cleared.")

