import re
from typing import List, Dict, Any, Optional

from app.models.chunk import Chunk, ChunkType
from app.models.retrieval import RetrievalResult
from app.storage.vector_store import VectorStore
from app.storage.bm25_store import BM25Store
from app.retrieval.query_expander import QueryExpander
from app.retrieval.exact_search import ExactSearchMatcher
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.context_expander import ContextExpander
from app.retrieval.evidence_validator import EvidenceValidator
from app.config.settings import CANDIDATE_POOL_SIZE


class HybridSearchEngine:

    def __init__(
        self,
        vector_store: VectorStore = None,
        bm25_store: BM25Store = None,
    ):
        self.vector_store = vector_store or VectorStore()
        self.bm25_store = bm25_store or BM25Store()

        self.query_expander = QueryExpander()
        self.exact_matcher = ExactSearchMatcher()
        self.reranker = CrossEncoderReranker()
        self.validator = EvidenceValidator()

    def search(
        self,
        query: str,
        n_results: int = 5,
        document_id: Optional[str] = None,
        tender_id: Optional[str] = None,
    ) -> List[RetrievalResult]:

        query_model = self.query_expander.expand_query(
            query,
            document_id=document_id,
            tender_id=tender_id,
        )

        # ---------------------------------------------------------
        # 1. Candidate retrieval
        # ---------------------------------------------------------

        vector_res = self.vector_store.query(
            query,
            n_results=CANDIDATE_POOL_SIZE,
            document_id=document_id,
            tender_id=tender_id,
        )

        v_ids = vector_res.get("ids", [[]])[0]
        v_docs = vector_res.get("documents", [[]])[0]
        v_metas = vector_res.get("metadatas", [[]])[0]
        v_dists = vector_res.get("distances", [[]])[0]

        bm25_scores = self.bm25_store.get_scores(
            query,
            document_id=document_id,
            tender_id=tender_id,
        )

        bm25_indices = sorted(
            enumerate(bm25_scores),
            key=lambda x: x[1],
            reverse=True,
        )[:CANDIDATE_POOL_SIZE]

        candidates: Dict[str, Dict[str, Any]] = {}

        def key(meta, text):
            return (
                f"{meta.get('source_doc','')}_"
                f"{meta.get('page_number',1)}_"
                f"{meta.get('char_start',0)}_"
                f"{re.sub(r'\\s+', ' ', text[:80].lower())}"
            )

        # Vector candidates
        for cid, doc, meta, dist in zip(
            v_ids,
            v_docs,
            v_metas,
            v_dists,
        ):
            k = key(meta, doc)

            candidates[k] = {
                "cid": cid,
                "chunk_id": self._numeric_chunk_id(cid),
                "text": doc,
                "metadata": meta,
                "dist": dist,
                "bm25": 0.0,
            }

        # BM25 candidates
        for idx, score in bm25_indices:

            if idx >= len(self.bm25_store.chunks_cache):
                continue

            cached = self.bm25_store.chunks_cache[idx]
            meta = cached["metadata"]
            text = cached["text"]

            # Explicit isolation.
            if tender_id and str(meta.get("tender_id")) != str(tender_id):
                continue

            if document_id and str(meta.get("document_id")) != str(document_id):
                continue

            k = key(meta, text)

            if k in candidates:
                candidates[k]["bm25"] = max(
                    candidates[k]["bm25"],
                    score,
                )
            else:
                candidates[k] = {
                    "cid": cached["id"],
                    "chunk_id": idx,
                    "text": text,
                    "metadata": meta,
                    "dist": 1.0,
                    "bm25": score,
                }

        if not candidates:
            return []

        candidate_list = list(candidates.values())

        # ---------------------------------------------------------
        # 2. Cross encoder
        # ---------------------------------------------------------

        texts = [x["text"] for x in candidate_list]
        rerank_scores = self.reranker.compute_scores(query, texts)

        if not rerank_scores:
            rerank_scores = [0.0] * len(candidate_list)

        min_rr = min(rerank_scores)
        max_rr = max(rerank_scores)
        rr_range = max(max_rr - min_rr, 1e-9)

        max_bm25 = max(
            [x["bm25"] for x in candidate_list],
            default=1.0,
        )
        max_bm25 = max(max_bm25, 1e-9)

        results = []

        # ---------------------------------------------------------
        # 3. Score retrieval relevance
        # ---------------------------------------------------------

        for i, cand in enumerate(candidate_list):

            meta = cand["metadata"]
            text = cand["text"]

            vector_similarity = 1.0 - max(
                0.0,
                min(1.0, cand["dist"]),
            )

            bm25_norm = cand["bm25"] / max_bm25

            exact = self.exact_matcher.compute_exact_score(
                query,
                text,
            )

            rr_norm = (
                rerank_scores[i] - min_rr
            ) / rr_range

            # Retrieval relevance ONLY.
            retrieval_score = (
                0.50 * rr_norm
                + 0.30 * bm25_norm
                + 0.20 * exact
            )

            # -----------------------------------------------------
            # 4. Independent evidence gate
            # -----------------------------------------------------

            answerable, confidence = (
                self.validator.evaluate_answerability(
                    query=query,
                    requested_attribute=query_model.requested_attribute,
                    doc_text=text,
                    score=retrieval_score,
                )
            )

            # -----------------------------------------------------
            # CRITICAL:
            #
            # Answerability does NOT get mixed into retrieval score.
            #
            # A highly relevant but non-answering chunk stays
            # NON-ANSWERABLE.
            # -----------------------------------------------------

            final_score = retrieval_score

            if not answerable:
                final_score -= 0.75

            chunk = Chunk(
                chunk_id=cand["chunk_id"],
                text=text,
                chunk_type=ChunkType(
                    meta.get("chunk_type", "text")
                ),
                page_number=int(
                    meta.get("page_number", 1)
                ),
                char_start=int(
                    meta.get("char_start", 0)
                ),
                char_end=int(
                    meta.get("char_end", len(text))
                ),
                document_id=str(
                    meta.get("document_id", "doc")
                ),
                source_doc=str(
                    meta.get("source_doc", "Unknown")
                ),
                tender_id=str(
                    meta.get("tender_id", "default_tender")
                ),
                document_type=str(
                    meta.get("document_type", "OTHER")
                ),
                document_version=int(
                    meta.get("document_version", 1)
                ),
                section=meta.get("section"),
                clause=meta.get("clause"),
                structured_json=meta.get(
                    "structured_json",
                    "{}",
                ),
                confidence=confidence,
            )

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    combined_score=final_score,
                    vector_score=vector_similarity,
                    bm25_score=bm25_norm,
                    exact_match_score=exact,
                    rerank_score=rerank_scores[i],
                    confidence=confidence,
                )
            )

        # ---------------------------------------------------------
        # 5. Sort
        # ---------------------------------------------------------

        results.sort(
            key=lambda r: r.combined_score,
            reverse=True,
        )

        # ---------------------------------------------------------
        # 6. Prefer answerable evidence
        #
        # BUT only after retrieval ranking.
        # ---------------------------------------------------------

        answerable_results = [
            r for r in results
            if "🟢" in r.confidence or "🟡" in r.confidence
        ]

        non_answerable_results = [
            r for r in results
            if "🔴" in r.confidence
        ]

        final_results = (
            answerable_results[:n_results]
            if answerable_results
            else non_answerable_results[:n_results]
        )

        # ---------------------------------------------------------
        # 7. Expand neighboring context
        # ---------------------------------------------------------

        top_dicts = [
            {
                "text": r.chunk.text,
                "chunk_id": r.chunk.chunk_id,
                "metadata": r.chunk.to_metadata_dict(),
            }
            for r in final_results
        ]

        ContextExpander.expand_candidate_contexts(
            top_dicts,
            self.bm25_store.chunks_cache,
        )

        for result, expanded in zip(
            final_results,
            top_dicts,
        ):
            result.chunk.text = expanded["text"]

        return final_results

    @staticmethod
    def _numeric_chunk_id(cid):
        try:
            return int(str(cid).split("_")[-1])
        except Exception:
            return 0