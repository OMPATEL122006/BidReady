import os
import json
import math
import re
from src.config import BM25_INDEX_PATH, STOP_WORDS

class BM25:
    def __init__(self, chunks=None, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lengths = []
        self.doc_freqs = {}
        self.idf = {}
        
        if chunks:
            self.fit(chunks)
            
    def _tokenize(self, text: str) -> list:
        """
        Tokenizes text by standardizing abbreviations, lowercasing,
        splitting on alphanumeric boundaries, and filtering out stop words.
        """
        text = text.lower()
        # Normalize common CPWD abbreviations to standard keywords
        text = text.replace("n.i.t.no.", "nit no").replace("n.i.t. no.", "nit no")
        text = text.replace("n.i.t. no", "nit no").replace("n.i.t.", "nit")
        text = text.replace("e.m.d.", "emd")
        
        words = re.findall(r'\b\w+\b', text)
        return [w for w in words if w not in STOP_WORDS]

    def fit(self, chunks: list):
        """
        Computes BM25 parameters across all text chunks.
        
        Args:
            chunks (list): List of dictionaries, each containing 'chunk_id' and 'text'.
        """
        self.corpus_size = len(chunks)
        self.doc_lengths = []
        self.doc_freqs = {}
        self.idf = {}
        
        # Temporary storage for term frequencies in each document
        doc_term_counts = []
        total_len = 0
        
        for c in chunks:
            tokens = self._tokenize(c["text"])
            self.doc_lengths.append(len(tokens))
            total_len += len(tokens)
            
            # Count terms in this document
            counts = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            doc_term_counts.append(counts)
            
            # Increment document frequencies (how many docs contain each term)
            for token in counts.keys():
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                
        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0.0
        
        # Precompute IDF for all terms
        for term, df in self.doc_freqs.items():
            # Standard BM25 IDF formulation (with a floor of 1e-5 to avoid negative IDFs for very common terms)
            num = self.corpus_size - df + 0.5
            denom = df + 0.5
            self.idf[term] = max(1e-5, math.log(1.0 + num / denom))
            
        # Store term frequency per doc for fast scoring
        self.doc_term_counts = doc_term_counts

    def get_scores(self, query: str) -> list:
        """
        Computes BM25 scores for the query against all fit documents.
        
        Returns:
            list: List of BM25 scores matching the indices of the fit chunks.
        """
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size
        
        if not query_tokens or self.corpus_size == 0:
            return scores
            
        for doc_idx in range(self.corpus_size):
            doc_len = self.doc_lengths[doc_idx]
            term_counts = self.doc_term_counts[doc_idx]
            score = 0.0
            
            for token in query_tokens:
                if token not in self.idf:
                    continue
                tf = term_counts.get(token, 0)
                if tf == 0:
                    continue
                    
                # BM25 scoring formula
                idf_val = self.idf[token]
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf_val * (num / denom)
                
            scores[doc_idx] = score
            
        return scores

    def save(self, filepath: str = BM25_INDEX_PATH):
        """
        Saves the fitted BM25 model variables to a JSON file.
        """
        data = {
            "k1": self.k1,
            "b": self.b,
            "corpus_size": self.corpus_size,
            "avg_doc_len": self.avg_doc_len,
            "doc_lengths": self.doc_lengths,
            "doc_freqs": self.doc_freqs,
            "idf": self.idf,
            "doc_term_counts": self.doc_term_counts
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, filepath: str = BM25_INDEX_PATH):
        """
        Loads the fitted BM25 model variables from a JSON file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"BM25 index not found at: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.k1 = data["k1"]
        self.b = data["b"]
        self.corpus_size = data["corpus_size"]
        self.avg_doc_len = data["avg_doc_len"]
        self.doc_lengths = data["doc_lengths"]
        self.doc_freqs = data["doc_freqs"]
        self.idf = data["idf"]
        self.doc_term_counts = data["doc_term_counts"]
