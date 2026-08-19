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
    print("  BIDREADY MATCHING EVIDENCE RANK DIAGNOSTIC (VECTOR/BM25/RERANK) ")
    print("=================================================================\n")

    available_files = scan_available_files()
    shibpur_files = [f for f in available_files if "shibpur" in f.lower()]

    if not shibpur_files:
        print("Error: Shibpur tender files not found in workspace.")
        return

    pipeline = TenderRAGPipeline()
    pipeline.clear()

    tender_id = pipeline.ingest_tender_set(shibpur_files)

    gt_map = {item.question: item for item in SHIBPUR_GROUND_TRUTH}

    output_lines = []
    output_lines.append("=================================================================")
    output_lines.append("   EVIDENCE RANK & SCORE BREAKDOWN ACROSS RETRIEVAL STAGES      ")
    output_lines.append("=================================================================\n")

    not_top1_count = 0
    found_count = 0

    for idx, question in enumerate(QUESTION_SUITE, 1):
        gt_item = gt_map.get(question)
        if gt_item and gt_item.expected_keyphrases:
            keywords = gt_item.expected_keyphrases
        else:
            stopwords = {"what", "is", "the", "of", "for", "this", "do", "we", "need", "in", "a", "an", "to", "how", "much", "are", "on", "at", "by", "or", "and", "be", "with", "from", "which", "required", "applicable", "allowed"}
            keywords = [w for w in question.lower().replace("?", "").split() if w not in stopwords and len(w) > 3][:3]

        # 1. Fetch Vector Store ordering & scores
        vector_res = pipeline.search_engine.vector_store.query(question, n_results=50, tender_id=tender_id)
        v_ids = vector_res.get("ids", [[]])[0]
        v_docs = vector_res.get("documents", [[]])[0]
        v_metas = vector_res.get("metadatas", [[]])[0]
        v_dists = vector_res.get("distances", [[]])[0]

        vector_ranks = {}
        vector_scores = {}
        for vr_i, (cid, doc, dist) in enumerate(zip(v_ids, v_docs, v_dists), 1):
            c_id = pipeline.search_engine._numeric_chunk_id(cid)
            vector_ranks[c_id] = vr_i
            vector_scores[c_id] = 1.0 - max(0.0, min(1.0, dist))

        # 2. Fetch BM25 Store ordering & scores
        bm25_scores = pipeline.search_engine.bm25_store.get_scores(question, tender_id=tender_id)
        bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)

        max_b25 = max(bm25_scores) if bm25_scores else 1.0
        if max_b25 <= 0.0:
            max_b25 = 1.0

        bm25_ranks = {}
        bm25_norm_scores = {}
        for br_i, (c_id, score) in enumerate(bm25_ranked, 1):
            bm25_ranks[c_id] = br_i
            bm25_norm_scores[c_id] = score / max_b25

        # 3. Full pipeline search (returns Final Ranked Top Candidates)
        final_results = pipeline.search_engine.search(question, n_results=10, tender_id=tender_id)

        # Build map of chunk_id -> final_results item
        final_ranks = {}
        final_objs = {}
        for fr_i, r in enumerate(final_results, 1):
            c_id = r.chunk.chunk_id
            final_ranks[c_id] = fr_i
            final_objs[c_id] = r

        # 4. Identify matching chunks containing evidence keywords
        matching_chunks = []
        for c_id, r_obj in final_objs.items():
            text_lower = r_obj.chunk.text.lower()
            if any(kp.lower() in text_lower for kp in keywords):
                matching_chunks.append(c_id)

        header_str = f"--- Q{idx}/{len(QUESTION_SUITE)}: {question} ---"
        kw_str = f"Keywords: {keywords}"
        
        output_lines.append(header_str)
        output_lines.append(kw_str)
        print(header_str)
        print(kw_str)

        if not matching_chunks:
            no_str = "  Status: NO MATCHING EVIDENCE IN RETRIEVED POOL"
            output_lines.append(no_str)
            print(no_str)
        else:
            found_count += 1
            best_matching_rank = min([final_ranks.get(cid, 999) for cid in matching_chunks])
            is_top1 = (best_matching_rank == 1)

            if not is_top1:
                not_top1_count += 1
                status_hdr = f"  Status: MATCHING EVIDENCE FOUND (BEST FINAL RANK: #{best_matching_rank} - NOT TOP-1! ⚠️)"
            else:
                status_hdr = f"  Status: MATCHING EVIDENCE FOUND (BEST FINAL RANK: #1 - TOP-1! ✅)"

            output_lines.append(status_hdr)
            print(status_hdr)

            for cid in matching_chunks:
                r_obj = final_objs[cid]
                v_rank = vector_ranks.get(cid, "N/A")
                v_score = vector_scores.get(cid, 0.0)
                b_rank = bm25_ranks.get(cid, "N/A")
                b_score = bm25_norm_scores.get(cid, 0.0)
                r_score = r_obj.rerank_score
                f_rank = final_ranks.get(cid, "N/A")
                f_score = r_obj.combined_score

                chunk_line = (
                    f"    Chunk ID: {cid} | Final Rank: #{f_rank} (Score: {f_score:.4f}) | "
                    f"Vector Rank: #{v_rank} (Score: {v_score:.4f}) | "
                    f"BM25 Rank: #{b_rank} (Score: {b_score:.4f}) | "
                    f"Reranker Raw Score: {r_score:.4f}"
                )
                txt_line = f"      Text: {r_obj.chunk.text.replace('\n', ' ').strip()[:160]}..."

                output_lines.append(chunk_line)
                output_lines.append(txt_line)
                print(chunk_line)
                print(txt_line)

        output_lines.append("-" * 65 + "\n")
        print("-" * 65 + "\n")

    summary_str = f"Diagnostic Summary: Evidence Found for {found_count}/70 questions. NOT TOP-1 in {not_top1_count} questions."
    print(summary_str)
    output_lines.append(summary_str)

    out_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "evidence_rank_diagnostic_report.txt")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"\n[DONE] Rank diagnostic report saved to: {out_file}")

if __name__ == "__main__":
    main()
