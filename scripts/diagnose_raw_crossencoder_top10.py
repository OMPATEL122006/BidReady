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
    print("   BIDREADY A/B DIAGNOSTIC: RAW CROSS-ENCODER TOP-10 REPORT     ")
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
    output_lines.append("   RAW CROSS-ENCODER CANDIDATE ORDERING A/B DIAGNOSTIC REPORT   ")
    output_lines.append("=================================================================\n")

    for idx, question in enumerate(QUESTION_SUITE, 1):
        gt_item = gt_map.get(question)
        if gt_item and gt_item.expected_keyphrases:
            keywords = gt_item.expected_keyphrases
        else:
            stopwords = {"what", "is", "the", "of", "for", "this", "do", "we", "need", "in", "a", "an", "to", "how", "much", "are", "on", "at", "by", "or", "and", "be", "with", "from", "which", "required", "applicable", "allowed"}
            keywords = [w for w in question.lower().replace("?", "").split() if w not in stopwords and len(w) > 3][:3]

        # Retrieve top-10 using raw cross-encoder score ordering
        top10_results = pipeline.search_engine.search(question, n_results=10, tender_id=tender_id)

        # Check candidate pool (vector + BM25) for existence of evidence
        v_res = pipeline.search_engine.vector_store.query(question, n_results=20, tender_id=tender_id)
        v_docs = v_res.get("documents", [[]])[0]
        v_metas = v_res.get("metadatas", [[]])[0]
        b_scores = pipeline.search_engine.bm25_store.get_scores(question, tender_id=tender_id)
        b_indices = sorted(enumerate(b_scores), key=lambda x: x[1], reverse=True)[:20]

        pool_texts = list(v_docs)
        for b_i, s in b_indices:
            if b_i < len(pipeline.search_engine.bm25_store.chunks_cache):
                pool_texts.append(pipeline.search_engine.bm25_store.chunks_cache[b_i]["text"])

        exists_in_pool = any(any(kp.lower() in t.lower() for kp in keywords) for t in pool_texts)

        # Find rank of first answerable evidence in top-10
        first_evidence_rank = "N/A"
        for rank, r in enumerate(top10_results, 1):
            text_lower = r.chunk.text.lower()
            if any(kp.lower() in text_lower for kp in keywords):
                first_evidence_rank = f"#{rank}"
                break

        top10_scores = [round(r.rerank_score, 4) for r in top10_results]
        top10_doc_pages = [f"{r.chunk.source_doc} (P{r.chunk.page_number})" for r in top10_results]

        header_str = f"--- Q{idx}/{len(QUESTION_SUITE)}: {question} ---"
        q_line1 = f"  1. Rank of first answerable evidence           : {first_evidence_rank}"
        q_line2 = f"  2. Answerable evidence exists in candidate pool: {'YES' if exists_in_pool else 'NO'}"
        q_line3 = f"  3. Top-10 raw reranker scores                  : {top10_scores}"
        q_line4 = f"  4. Top-10 document/page IDs                    : {top10_doc_pages}"

        print(header_str)
        print(q_line1)
        print(q_line2)
        print(q_line3)
        print(q_line4)
        print("-" * 65 + "\n")

        output_lines.append(header_str)
        output_lines.append(q_line1)
        output_lines.append(q_line2)
        output_lines.append(q_line3)
        output_lines.append(q_line4)
        output_lines.append("-" * 65 + "\n")

    out_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "raw_crossencoder_ab_diagnostic_report.txt")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"\n[DONE] Raw Cross-Encoder A/B Diagnostic complete! Saved to: {out_file}")

if __name__ == "__main__":
    main()
