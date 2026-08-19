import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.pipeline import TenderRAGPipeline
from app.evaluation.ground_truth import SHIBPUR_GROUND_TRUTH
from scripts.validate_retrieval import QUESTION_SUITE, scan_available_files

def main():
    print("=================================================================")
    print("  BIDREADY CANDIDATE POOL EVIDENCE DIAGNOSTIC (PRE-RANKING)      ")
    print("=================================================================\n")

    available_files = scan_available_files()
    shibpur_files = [f for f in available_files if "shibpur" in f.lower()]

    if not shibpur_files:
        print("Error: Shibpur tender files not found in workspace.")
        return

    pipeline = TenderRAGPipeline()
    pipeline.clear()

    tender_id = pipeline.ingest_tender_set(shibpur_files)

    # Build lookup map for ground truth keyphrases
    gt_map = {item.question: item for item in SHIBPUR_GROUND_TRUTH}

    output_lines = []
    output_lines.append("=================================================================")
    output_lines.append("  PRE-RANKING RETRIEVAL CANDIDATE POOL DIAGNOSTIC REPORT         ")
    output_lines.append("=================================================================\n")

    for idx, question in enumerate(QUESTION_SUITE, 1):
        gt_item = gt_map.get(question)
        if gt_item and gt_item.expected_keyphrases:
            keywords = gt_item.expected_keyphrases
        else:
            # Fallback keyphrases from query intent for questions without explicit ground truth item
            stopwords = {"what", "is", "the", "of", "for", "this", "do", "we", "need", "in", "a", "an", "to", "how", "much", "are", "on", "at", "by", "or", "and", "be", "with", "from", "which", "required", "applicable", "allowed"}
            keywords = [w for w in question.lower().replace("?", "").split() if w not in stopwords and len(w) > 3][:3]

        # Fetch candidate pool (vector + BM25) BEFORE final ranking/reranking
        query_model = pipeline.search_engine.query_expander.expand_query(question, tender_id=tender_id)
        
        # Candidate fetch directly from vector & BM25 stores (Candidate Pool before reranker)
        vector_res = pipeline.search_engine.vector_store.query(question, n_results=20, tender_id=tender_id)
        v_docs = vector_res.get("documents", [[]])[0]
        v_metas = vector_res.get("metadatas", [[]])[0]
        v_ids = vector_res.get("ids", [[]])[0]

        bm25_scores = pipeline.search_engine.bm25_store.get_scores(question, tender_id=tender_id)
        bm25_indices = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:20]

        candidate_chunks = {}
        for cid, doc, meta in zip(v_ids, v_docs, v_metas):
            c_id = pipeline.search_engine._numeric_chunk_id(cid)
            candidate_chunks[c_id] = {"chunk_id": c_id, "text": doc, "metadata": meta}

        for idx_b, score in bm25_indices:
            if idx_b < len(pipeline.search_engine.bm25_store.chunks_cache):
                cached = pipeline.search_engine.bm25_store.chunks_cache[idx_b]
                c_id = idx_b
                if c_id not in candidate_chunks:
                    candidate_chunks[c_id] = {"chunk_id": c_id, "text": cached["text"], "metadata": cached["metadata"]}

        matching_chunk_ids = []
        for c_id, cand in candidate_chunks.items():
            text_lower = cand["text"].lower()
            # Match if any keyphrase appears in chunk text
            if any(kp.lower() in text_lower for kp in keywords):
                matching_chunk_ids.append(c_id)

        status_str = "FOUND" if matching_chunk_ids else "NOT FOUND"
        kw_str = ", ".join([f"'{k}'" for k in keywords])
        ids_str = ", ".join([str(c) for c in sorted(matching_chunk_ids)]) if matching_chunk_ids else "None"

        line = f"Q{idx}, expected evidence keywords: [{kw_str}], {status_str}, matching chunk IDs: [{ids_str}]"
        print(line)
        output_lines.append(line)

    out_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "pre_ranking_candidate_pool_diagnostic.txt")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"\n[DONE] Pre-ranking candidate pool diagnostic complete! Saved to: {out_file}")

if __name__ == "__main__":
    main()
