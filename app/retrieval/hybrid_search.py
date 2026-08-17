import math
import re
from typing import List, Dict, Any, Optional
from app.models.chunk import Chunk, ChunkType
from app.models.retrieval import RetrievalResult, QueryModel
from app.storage.vector_store import VectorStore
from app.storage.bm25_store import BM25Store
from app.retrieval.query_expander import QueryExpander
from app.retrieval.exact_search import ExactSearchMatcher
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.context_expander import ContextExpander
from app.retrieval.evidence_validator import EvidenceValidator
from app.config.settings import CANDIDATE_POOL_SIZE
from app.config.logging import logger

class HybridSearchEngine:
    """
    Two-stage retrieval and reranking engine:
    1. Candidate Retrieval & Strict Deduplication (Vector Store + BM25 Lexical Store)
    2. Primary Reranking via Cross-Encoder with Pool-Level Min-Max Normalization
    3. Dynamic Exact Entity Matching & Generic Target Validation
    4. Context Expansion
    """
    def __init__(self, vector_store: VectorStore = None, bm25_store: BM25Store = None):
        self.vector_store = vector_store or VectorStore()
        self.bm25_store = bm25_store or BM25Store()
        self.query_expander = QueryExpander()
        self.exact_matcher = ExactSearchMatcher()
        self.reranker = CrossEncoderReranker()
        self.validator = EvidenceValidator()

    def search(self, query: str, n_results: int = 3, document_id: Optional[str] = None, tender_id: Optional[str] = None) -> List[RetrievalResult]:
        # 1. Expand query & classify target intent
        query_model = self.query_expander.expand_query(query, document_id=document_id, tender_id=tender_id)

        # 2. Fetch Vector candidates
        vector_res = self.vector_store.query(query, n_results=CANDIDATE_POOL_SIZE, document_id=document_id, tender_id=tender_id)
        v_ids = vector_res["ids"][0] if vector_res and vector_res.get("ids") else []
        v_docs = vector_res["documents"][0] if vector_res and vector_res.get("documents") else []
        v_metas = vector_res["metadatas"][0] if vector_res and vector_res.get("metadatas") else []
        v_dists = vector_res["distances"][0] if vector_res and vector_res.get("distances") else []

        # 3. Fetch BM25 Lexical scores
        bm25_scores = self.bm25_store.get_scores(query, document_id=document_id, tender_id=tender_id)
        bm25_scored_indices = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:CANDIDATE_POOL_SIZE]

        # 4. Strict Deduplication by Canonical Content / Metadata Key
        candidates: Dict[str, Dict[str, Any]] = {}

        def get_chunk_key(doc_name: str, p_num: int, c_start: int, txt: str) -> str:
            clean_t = re.sub(r'\s+', ' ', txt.strip().lower()[:100])
            return f"{doc_name}_p{p_num}_s{c_start}_{clean_t}"

        # Add Vector candidates
        for cid, doc, meta, dist in zip(v_ids, v_docs, v_metas, v_dists):
            source_doc = str(meta.get("source_doc", "Unknown"))
            page_num = int(meta.get("page_number", 1))
            c_start = int(meta.get("char_start", 0))
            ckey = get_chunk_key(source_doc, page_num, c_start, doc)

            chunk_num_id = int(cid.split("_")[-1]) if "_" in cid and cid.split("_")[-1].isdigit() else 0
            candidates[ckey] = {
                "cid": cid,
                "chunk_id": chunk_num_id,
                "text": doc,
                "metadata": meta,
                "dist": dist,
                "bm25_score": 0.0
            }

        # Merge BM25 candidates
        for idx, score in bm25_scored_indices:
            if idx < len(self.bm25_store.chunks_cache):
                c_cache = self.bm25_store.chunks_cache[idx]
                meta = c_cache["metadata"]
                doc_text = c_cache["text"]
                source_doc = str(meta.get("source_doc", "Unknown"))
                page_num = int(meta.get("page_number", 1))
                c_start = int(meta.get("char_start", 0))
                ckey = get_chunk_key(source_doc, page_num, c_start, doc_text)

                if ckey in candidates:
                    candidates[ckey]["bm25_score"] = max(candidates[ckey]["bm25_score"], score)
                else:
                    candidates[ckey] = {
                        "cid": c_cache["id"],
                        "chunk_id": idx,
                        "text": doc_text,
                        "metadata": meta,
                        "dist": 1.0,
                        "bm25_score": score
                    }

        cand_list = list(candidates.values())
        if not cand_list:
            return []

        # 5. Cross-Encoder Reranking
        doc_texts = [c["text"] for c in cand_list]
        reranker_scores = self.reranker.compute_scores(query, doc_texts)

        # Min-Max Normalization of Cross-Encoder Scores across current candidate pool
        if reranker_scores:
            min_rr = min(reranker_scores)
            max_rr = max(reranker_scores)
            rr_range = max_rr - min_rr if max_rr > min_rr else 1.0
        else:
            min_rr, rr_range = 0.0, 1.0

        max_bm25 = max([c["bm25_score"] for c in cand_list]) if cand_list else 1.0
        if max_bm25 <= 0.0:
            max_bm25 = 1.0

        scored_candidates = []
        for idx, cand in enumerate(cand_list):
            doc_text = cand["text"]
            dist = cand["dist"]
            sim = 1.0 - max(0.0, min(1.0, dist))
            norm_bm25 = cand["bm25_score"] / max_bm25
            meta = cand.get("metadata") or {}
            c_type = meta.get("chunk_type", "text")

            exact_score = self.exact_matcher.compute_exact_score(query, doc_text)
            domain_boost = self.validator.compute_domain_boost(query, doc_text)
            boq_penalty = self.validator.compute_boq_penalty(query, c_type, doc_text)

            rr_score = reranker_scores[idx]

            if self.reranker.enabled:
                # Relative Cross-Encoder Score (0.0 to 1.0)
                norm_rerank = (rr_score - min_rr) / rr_range
                # Cross-encoder is the PRIMARY ranker (60%), exact entity match (20%), BM25 baseline (20%)
                raw_score = 0.60 * norm_rerank + 0.20 * exact_score + 0.20 * norm_bm25 + domain_boost + boq_penalty
            else:
                raw_score = 0.40 * sim + 0.40 * norm_bm25 + 0.20 * exact_score + domain_boost + boq_penalty

            # Run Evidence Answerability Check (Directly Answers vs Merely Related)
            is_answerable, conf = self.validator.evaluate_answerability(
                query=query,
                requested_attribute=query_model.requested_attribute,
                doc_text=doc_text,
                score=raw_score
            )

            # Severe penalty for non-answering distractor chunks
            if not is_answerable:
                final_score = raw_score - 1.5
            else:
                final_score = raw_score

            c_obj = Chunk(
                chunk_id=cand["chunk_id"],
                text=doc_text,
                chunk_type=ChunkType(meta.get("chunk_type", "text")),
                page_number=int(meta.get("page_number", 1)),
                char_start=int(meta.get("char_start", 0)),
                char_end=int(meta.get("char_end", len(doc_text))),
                document_id=str(meta.get("document_id", "doc")),
                source_doc=str(meta.get("source_doc", "Unknown")),
                tender_id=str(meta.get("tender_id", "default_tender")),
                document_type=str(meta.get("document_type", "OTHER")),
                document_version=int(meta.get("document_version", 1)),
                section=meta.get("section"),
                clause=meta.get("clause"),
                structured_json=meta.get("structured_json", "{}"),
                confidence=conf
            )

            scored_candidates.append(RetrievalResult(
                chunk=c_obj,
                combined_score=final_score,
                vector_score=sim,
                bm25_score=norm_bm25,
                exact_match_score=exact_score,
                rerank_score=rr_score,
                confidence=conf
            ))

        # 6. Primary Ranking by Combined Score (Deduplicated & Cross-Encoder Calibrated)
        scored_candidates.sort(key=lambda x: x.combined_score, reverse=True)

        # 7. Context Expansion on Top Candidates
        top_candidates = scored_candidates[:n_results]
        top_dicts = [{"text": r.chunk.text, "chunk_id": r.chunk.chunk_id, "metadata": r.chunk.to_metadata_dict()} for r in top_candidates]
        ContextExpander.expand_candidate_contexts(top_dicts, self.bm25_store.chunks_cache)

        for r, td in zip(top_candidates, top_dicts):
            r.chunk.text = td["text"]

        return top_candidates
