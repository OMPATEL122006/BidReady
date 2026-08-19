import os
import sys
import re

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.pipeline import TenderRAGPipeline
from scripts.validate_retrieval import QUESTION_SUITE, scan_available_files

TARGET_Q_INDICES = [1, 5, 24, 31, 35, 40, 45, 57]

def main():
    print("=================================================================")
    print(" BIDREADY SCORING COMPONENT DIAGNOSTIC (FORMULA & VALUE BREAKDOWN)")
    print(" Target Questions: Q1, Q5, Q24, Q31, Q35, Q40, Q45, Q57         ")
    print("=================================================================\n")

    available_files = scan_available_files()
    shibpur_files = [f for f in available_files if "shibpur" in f.lower()]

    if not shibpur_files:
        print("Error: Shibpur tender files not found in workspace.")
        return

    pipeline = TenderRAGPipeline()
    pipeline.clear()

    tender_id = pipeline.ingest_tender_set(shibpur_files)

    output_lines = []
    output_lines.append("=================================================================")
    output_lines.append("   DETAILED SCORING FORMULA & COMPONENT BREAKDOWN REPORT       ")
    output_lines.append("   Formula: final_score = (0.50 * rr_norm) + (0.20 * exact) +  ")
    output_lines.append("                          (0.30 * bm25_norm) + domain_boost +    ")
    output_lines.append("                          boq_penalty + validator_contrib        ")
    output_lines.append("=================================================================\n")

    for q_idx in TARGET_Q_INDICES:
        question = QUESTION_SUITE[q_idx - 1]
        q_model = pipeline.search_engine.query_expander.expand_query(question, tender_id=tender_id)

        header_str = f"=== Q{q_idx}: {question} ==="
        output_lines.append(header_str)
        print(header_str)

        # Retrieve top 5 using search_engine to inspect exact candidate scoring components
        top_results = pipeline.search_engine.search(question, n_results=5, tender_id=tender_id)

        # Re-fetch candidate list from search engine logic to extract exact intermediate terms
        vector_res = pipeline.search_engine.vector_store.query(question, n_results=20, tender_id=tender_id)
        v_ids = vector_res.get("ids", [[]])[0]
        v_docs = vector_res.get("documents", [[]])[0]
        v_metas = vector_res.get("metadatas", [[]])[0]
        v_dists = vector_res.get("distances", [[]])[0]

        bm25_scores = pipeline.search_engine.bm25_store.get_scores(question, tender_id=tender_id)
        bm25_indices = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:20]

        candidates = {}
        def key(meta, text):
            return f"{meta.get('source_doc','')}_{meta.get('page_number',1)}_{meta.get('char_start',0)}_{re.sub(r'\\s+', ' ', text[:80].lower())}"

        for cid, doc, meta, dist in zip(v_ids, v_docs, v_metas, v_dists):
            k = key(meta, doc)
            candidates[k] = {"cid": cid, "chunk_id": pipeline.search_engine._numeric_chunk_id(cid), "text": doc, "metadata": meta, "dist": dist, "bm25": 0.0}

        for idx_b, score in bm25_indices:
            if idx_b < len(pipeline.search_engine.bm25_store.chunks_cache):
                cached = pipeline.search_engine.bm25_store.chunks_cache[idx_b]
                meta = cached["metadata"]
                text = cached["text"]
                k = key(meta, text)
                if k in candidates:
                    candidates[k]["bm25"] = max(candidates[k]["bm25"], score)
                else:
                    candidates[k] = {"cid": cached["id"], "chunk_id": idx_b, "text": text, "metadata": meta, "dist": 1.0, "bm25": score}

        cand_list = list(candidates.values())
        texts = [x["text"] for x in cand_list]
        rerank_scores = pipeline.search_engine.reranker.compute_scores(question, texts)
        if not rerank_scores:
            rerank_scores = [0.0] * len(cand_list)

        min_rr = min(rerank_scores)
        max_rr = max(rerank_scores)
        rr_range = max(max_rr - min_rr, 1e-9)

        max_bm25 = max([x["bm25"] for x in cand_list], default=1.0)
        max_bm25 = max(max_bm25, 1e-9)

        # Match top_results to cand_list items to print component breakdown
        for rank, r in enumerate(top_results, 1):
            c_id = r.chunk.chunk_id
            text = r.chunk.text
            
            # Find candidate item in cand_list
            cand_item = next((c for c in cand_list if c["chunk_id"] == c_id), None)
            idx_in_list = cand_list.index(cand_item) if cand_item else 0
            
            raw_rr = rerank_scores[idx_in_list] if idx_in_list < len(rerank_scores) else r.rerank_score
            rr_norm = (raw_rr - min_rr) / rr_range
            bm25_norm = r.bm25_score
            exact_raw = r.exact_match_score

            reranker_contrib = 0.50 * rr_norm
            exact_contrib = 0.20 * exact_raw
            bm25_contrib = 0.30 * bm25_norm
            domain_boost = 0.0
            boq_penalty = 0.0

            retrieval_score = reranker_contrib + exact_contrib + bm25_contrib + domain_boost + boq_penalty

            ans, conf = pipeline.search_engine.validator.evaluate_answerability(
                query=question,
                requested_attribute=q_model.requested_attribute,
                doc_text=text,
                score=retrieval_score
            )

            validator_contrib = 0.0 if ans else -0.75
            final_calculated = retrieval_score + validator_contrib

            c_info = (
                f"\n  [Candidate #{rank}] Chunk ID: {c_id} | Doc: {r.chunk.source_doc} (P{r.chunk.page_number}) | Confidence: {r.confidence}\n"
                f"    Formula: final_score = reranker_contrib + exact_contrib + bm25_contrib + domain_boost + boq_penalty + validator_contrib\n"
                f"    ---------------------------------------------------------------------------------------------------------\n"
                f"    - reranker contribution          : {reranker_contrib:.4f}  (0.50 * rr_norm: {rr_norm:.4f}, raw logit: {raw_rr:.4f})\n"
                f"    - exact-match contribution       : {exact_contrib:.4f}  (0.20 * exact_raw: {exact_raw:.4f})\n"
                f"    - BM25 contribution              : {bm25_contrib:.4f}  (0.30 * bm25_norm: {bm25_norm:.4f})\n"
                f"    - domain boost                   : {domain_boost:.4f}\n"
                f"    - BOQ penalty                    : {boq_penalty:.4f}\n"
                f"    - answerability/validator contrib: {validator_contrib:.4f}  (answerable: {ans})\n"
                f"    = final_score                    : {final_calculated:.4f}\n"
                f"    Text snippet: {text.replace('\n', ' ').strip()[:160]}..."
            )

            output_lines.append(c_info)
            print(c_info)

        output_lines.append("\n" + "=" * 65 + "\n")
        print("\n" + "=" * 65 + "\n")

    out_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "scoring_breakdown_diagnostic_report.txt")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"\n[DONE] Scoring breakdown diagnostic report saved to: {out_file}")

if __name__ == "__main__":
    main()
