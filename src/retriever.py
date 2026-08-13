import re
import math
from src.config import CANDIDATE_POOL_SIZE, STOP_WORDS
from src.database import get_chroma_client, get_or_create_collection, query_chroma
from src.bm25 import BM25
from src.reranker import BGEReranker
from db_experiment import compute_keyword_score, compute_phrase_boost

class TenderRetriever:
    def __init__(self):
        # 1. Initialize Clients
        self.client = get_chroma_client()
        self.collection = get_or_create_collection(self.client, "tender_requirements")
        self.reranker = BGEReranker()
        
        # 2. Build BM25 Index Dynamically from database chunks
        self.chunks_cache = []
        self.bm25 = None
        self._initialize_bm25()

    def _initialize_bm25(self):
        """
        Fetches all stored chunks in ChromaDB and fits the BM25 model on them.
        """
        count = self.collection.count()
        if count == 0:
            print("Warning: Chroma collection is empty. Cannot initialize BM25 yet.")
            return
            
        print(f"Loading {count} chunks from Chroma to fit local BM25 index...")
        all_data = self.collection.get()
        
        chunks = []
        for cid, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
            chunks.append({
                "chunk_id": int(cid.split("_")[1]),
                "text": doc,
                "metadata": meta
            })
            
        # Sort chunks by ID to align indices
        chunks.sort(key=lambda x: x["chunk_id"])
        self.chunks_cache = chunks
        
        # Fit BM25
        self.bm25 = BM25(chunks)
        print("BM25 index successfully fitted.")

    def _get_domain_boost_score(self, query: str, doc_text: str) -> float:
        """
        Applies deterministic penalties and boosts based on exact terminology and values
        to prevent semantic confusion (e.g. confusing EMD with Performance Guarantee).
        """
        # Apply deterministic standardizations to query and doc text for matching rule intents
        q_lower = query.lower()
        q_lower = q_lower.replace("n.i.t.no.", "nit no").replace("n.i.t. no.", "nit no")
        q_lower = q_lower.replace("n.i.t. no", "nit no").replace("n.i.t.", "nit")
        q_lower = q_lower.replace("e.m.d.", "emd")

        d_lower = doc_text.lower()
        d_lower = d_lower.replace("n.i.t.no.", "nit no").replace("n.i.t. no.", "nit no")
        d_lower = d_lower.replace("n.i.t. no", "nit no").replace("n.i.t.", "nit")
        d_lower = d_lower.replace("e.m.d.", "emd")

        boost = 0.0

        # Amount and Date patterns
        currency_val_pattern = r'(?:rs\.?|₹|rupees?)\s*\d+[\d,]*'
        date_pattern = r'\d{2}\.\d{2}\.\d{4}'
        time_pattern = r'\d{2}\.\d{2}\s*(?:hrs|am|pm)'

        # --- RULE 1: NIT Number Search (Notice Inviting Tender number reference) ---
        if "nit number" in q_lower or "nit no" in q_lower:
            # Look for typical CPWD NIT number formats: e.g. 17/AE/PCSD/LKO/26-27 or slashes
            if re.search(r'\b\d+/[A-Z0-9/]+-\d+\b|\b\d+/[A-Z0-9/]+\b', d_lower):
                boost += 0.5
            # Strongly penalize National Institute of Technology (NIT) concrete design testing reference
            if any(w in d_lower for w in ["iit", "laboratories", "test house", "laboratory", "mix design", "institutes"]):
                boost -= 0.7

        # --- RULE 2: EMD (Earnest Money Deposit) ---
        elif "emd" in q_lower or "earnest money" in q_lower:
            # Check if query is looking for deposit forms vs deposit amount
            if any(w in q_lower for w in ["forms", "acceptable", "how to", "mode", "payment", "instrument", "deposited"]):
                # Boost if document lists acceptable EMD forms
                payment_keywords = ["demand draft", "pay order", "banker's cheque", "fixed deposit", "bank guarantee", "rtgs", "neft", "receipt of deposition"]
                matches = sum(1 for w in payment_keywords if w in d_lower)
                if matches >= 2:
                    boost += 0.5
                if "deposited" in d_lower or "deposited through" in d_lower:
                    boost += 0.3
            else:
                # Searching for EMD Amount
                if "earnest money" in d_lower or "emd" in d_lower:
                    boost += 0.3
                if re.search(currency_val_pattern, d_lower):
                    boost += 0.2
            # Penalize performance guarantee to avoid mixing them up
            if "performance guarantee" in d_lower or "security deposit" in d_lower:
                boost -= 0.6

        # --- RULE 3: Completion Period vs. Guarantee / Defect Liability Period ---
        elif any(k in q_lower for k in ["completion", "period of completion", "how long", "duration"]):
            if "completion" in d_lower and any(w in d_lower for w in ["month", "months", "days", "weeks"]):
                boost += 0.3
            # Strongly penalize chunks talking about maintenance/defect liability/tile guarantee
            if any(w in d_lower for w in ["guarantee bond", "stone", "tile", "defect", "liability", "warranty"]):
                boost -= 0.7

        # --- RULE 4: NIT Date ---
        elif "nit date" in q_lower or "date of the nit" in q_lower or "date of nit" in q_lower:
            # Look for N.I.T.No & Date format
            if "nit" in d_lower and "dated" in d_lower:
                boost += 0.6
            elif re.search(date_pattern, d_lower):
                boost += 0.2
            # Penalize generic cost estimate or page 1 Approval
            if "approved" in d_lower and "estimated cost" in d_lower:
                boost -= 0.4

        # --- RULE 5: Financial Cover Opening ---
        elif "financial cover" in q_lower and any(k in q_lower for k in ["opening", "opened"]):
            if "financial cover" in d_lower or "financial" in d_lower:
                boost += 0.3
            if any(w in d_lower for w in ["later date", "communicated", "notified later"]):
                boost += 0.4
            # Penalize eligibility cover opening dates to avoid confusion
            if "eligibility cover" in d_lower or "cover 1" in d_lower:
                boost -= 0.6

        # --- RULE 6: Eligibility Cover Bid Opening Date/Time seeking "When" ---
        elif any(k in q_lower for k in ["opening", "opened"]):
            # Check if query contains "when" (seeking date)
            if "when" in q_lower:
                if re.search(date_pattern, d_lower) or re.search(time_pattern, d_lower):
                    boost += 0.4
                else:
                    boost -= 0.5  # Penalize chunks with no date/time
            if "opening" in d_lower or "opened" in d_lower:
                boost += 0.3
            if "eligibility" in d_lower:
                boost += 0.2
            # Penalize submission rules
            if "submission" in d_lower:
                boost -= 0.3

        # --- RULE 7: Bid Submission Deadline ---
        elif any(k in q_lower for k in ["submission", "last date", "deadline"]):
            if "submission" in d_lower and "last date" in d_lower:
                boost += 0.3
            if re.search(date_pattern, d_lower):
                boost += 0.2
            # Penalize opening rules
            if "opening" in d_lower or "opened" in d_lower:
                boost -= 0.5

        # --- RULE 8: Similar Work ---
        elif "similar work" in q_lower:
            if "similar work means" in d_lower:
                boost += 0.5
            if "specialized" in d_lower:
                boost -= 0.3

        return boost

    def retrieve(self, query: str, n_results: int = 3, debug: bool = False) -> list:
        """
        Retrieves the top N matching chunks using a hybrid candidate retrieval
        followed by a Cross-Encoder semantic reranker and domain rule booster.
        """
        if not self.chunks_cache:
            # Try to initialize if it was empty earlier
            self._initialize_bm25()
            if not self.chunks_cache:
                return []

        # 1. Fetch Candidate Chunks
        # Vector candidates (ChromaDB)
        vector_res = query_chroma(self.collection, query, n_results=CANDIDATE_POOL_SIZE)
        v_ids = vector_res["ids"][0]
        v_docs = vector_res["documents"][0]
        v_metas = vector_res["metadatas"][0]
        v_dists = vector_res["distances"][0]

        # BM25 candidates
        bm25_scores = self.bm25.get_scores(query)
        bm25_scored_indices = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:CANDIDATE_POOL_SIZE]
        
        # Merge candidate pools (union of IDs to avoid duplicates)
        candidates = {}
        
        # Add vector candidates
        for cid, doc, meta, dist in zip(v_ids, v_docs, v_metas, v_dists):
            candidates[cid] = {
                "chunk_id": int(cid.split("_")[1]),
                "text": doc,
                "metadata": meta,
                "dist": dist,
                "bm25_score": 0.0  # will fill below
            }
            
        # Add BM25 candidates
        for idx, score in bm25_scored_indices:
            cid = f"chunk_{idx}"
            if cid not in candidates:
                candidates[cid] = {
                    "chunk_id": idx,
                    "text": self.chunks_cache[idx]["text"],
                    "metadata": self.chunks_cache[idx]["metadata"],
                    "dist": 1.0,  # default distance if not retrieved by vector search
                    "bm25_score": score
                }
            else:
                candidates[cid]["bm25_score"] = score

        # Fill BM25 scores for vector candidates that were not in top BM25
        for cid, cand in candidates.items():
            if cand["bm25_score"] == 0.0:
                cand["bm25_score"] = bm25_scores[cand["chunk_id"]]

        # 2. Normalized Scoring Components
        # Find max BM25 score for normalization
        max_bm25 = max([c["bm25_score"] for c in candidates.values()]) if candidates else 1.0
        if max_bm25 == 0.0:
            max_bm25 = 1.0
            
        cand_list = list(candidates.values())
        
        # 3. Compute Reranker Scores (if enabled)
        doc_texts = [c["text"] for c in cand_list]
        reranker_scores = [0.0] * len(cand_list)
        
        if self.reranker.enabled:
            reranker_scores = self.reranker.compute_scores(query, doc_texts)

        # 4. Integrate all scoring components
        scored_candidates = []
        for idx, cand in enumerate(cand_list):
            doc = cand["text"]
            dist = cand["dist"]
            
            # Normalize Vector Sim (1 = closest, 0 = furthest)
            sim = 1.0 - max(0.0, min(1.0, dist))
            
            # Normalize BM25
            norm_bm25 = cand["bm25_score"] / max_bm25
            
            # Phrase boost
            phrase_boost = compute_phrase_boost(query, doc)
            
            # Domain Rules boost / penalty
            domain_boost = self._get_domain_boost_score(query, doc)
            
            # Reranker score (BGE output range can vary, normalize if enabled)
            rerank_score = reranker_scores[idx]
            
            # Calculate final combined score
            if self.reranker.enabled:
                # Use Reranker score as core semantic metric, augmented by deterministic rules
                # BGE base outputs raw logits where values above 0 are generally relevant.
                # Shift and scale BGE score to ~ [0, 1] range for readable combined score
                norm_rerank = 1.0 / (1.0 + math.exp(-rerank_score)) # sigmoid normalization
                final_score = 0.5 * norm_rerank + 0.3 * norm_bm25 + 0.2 * phrase_boost + domain_boost
            else:
                # Hybrid fallback
                final_score = 0.4 * sim + 0.4 * norm_bm25 + 0.2 * phrase_boost + domain_boost
                
            scored_candidates.append({
                "chunk_id": cand["chunk_id"],
                "text": doc,
                "metadata": cand["metadata"],
                "dist": dist,
                "sim": sim,
                "bm25_score": cand["bm25_score"],
                "norm_bm25": norm_bm25,
                "phrase_boost": phrase_boost,
                "domain_boost": domain_boost,
                "rerank_score": rerank_score,
                "score": final_score
            })

        # Sort by final score descending
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        if debug:
            return scored_candidates
            
        return scored_candidates[:n_results]
