import os
import json
import math
import re
from typing import List, Dict, Any, Optional
from app.config.settings import BM25_INDEX_PATH, STOP_WORDS
from app.models.chunk import Chunk
from app.config.logging import logger

class BM25Store:
    """
    Local BM25 lexical indexer with document_id filtering support.
    """
    def __init__(self, index_path: str = BM25_INDEX_PATH, k1: float = 1.5, b: float = 0.75):
        self.index_path = index_path
        self.k1 = k1
        self.b = b
        self.chunks_cache: List[Dict[str, Any]] = []
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.corpus_size: int = 0

    def tokenize(self, text: str) -> List[str]:
        text_clean = text.lower()
        text_clean = text_clean.replace("n.i.t.no.", "nit no").replace("n.i.t. no.", "nit no")
        text_clean = text_clean.replace("n.i.t. no", "nit no").replace("n.i.t.", "nit")
        text_clean = text_clean.replace("e.m.d.", "emd").replace("g.s.t.", "gst")
        tokens = re.findall(r'\b[a-zA-Z0-9%\.]+\b', text_clean)
        return [t for t in tokens if t not in STOP_WORDS]

    def fit(self, chunks: List[Chunk]):
        self.chunks_cache = []
        self.doc_len = []
        self.doc_freqs = []
        self.corpus_size = len(chunks)

        if self.corpus_size == 0:
            return

        df_counts: Dict[str, int] = {}
        for idx, c in enumerate(chunks):
            c_dict = {
                "id": f"chunk_{c.source_doc}_{c.document_id}_{c.chunk_id}",
                "numeric_id": idx,
                "text": c.text,
                "metadata": c.to_metadata_dict()
            }
            self.chunks_cache.append(c_dict)
            tokens = self.tokenize(c.text)
            self.doc_len.append(len(tokens))

            freqs: Dict[str, int] = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_freqs.append(freqs)

            for t in freqs.keys():
                df_counts[t] = df_counts.get(t, 0) + 1

        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0.0

        self.idf = {}
        for term, df in df_counts.items():
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

        logger.info(f"BM25 index successfully fitted on {self.corpus_size} chunks.")

    def get_scores(self, query: str, document_id: Optional[str] = None, tender_id: Optional[str] = None) -> List[float]:
        q_tokens = self.tokenize(query)
        scores = [0.0] * self.corpus_size
        if self.corpus_size == 0:
            return scores

        for idx in range(self.corpus_size):
            meta = self.chunks_cache[idx]["metadata"]
            if tender_id:
                c_tender_id = str(meta.get("tender_id", ""))
                if c_tender_id != str(tender_id):
                    continue
            if document_id:
                c_doc_id = str(meta.get("document_id", ""))
                if c_doc_id != str(document_id):
                    continue

            d_len = self.doc_len[idx]
            d_freqs = self.doc_freqs[idx]
            score = 0.0

            for token in q_tokens:
                if token not in d_freqs:
                    continue
                freq = d_freqs[token]
                idf_val = self.idf.get(token, 0.0)
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (d_len / self.avgdl))
                score += idf_val * (numerator / denominator)

            scores[idx] = score

        return scores
