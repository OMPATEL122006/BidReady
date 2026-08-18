import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.pipeline import TenderRAGPipeline
from scripts.validate_retrieval import QUESTION_SUITE, scan_available_files

def main():
    print("=================================================================")
    print("   BIDREADY RETRIEVAL DIAGNOSTIC: TOP 5 CANDIDATES & SCORES     ")
    print("=================================================================\n")

    available_files = scan_available_files()
    shibpur_files = [f for f in available_files if "shibpur" in f.lower()]

    if not shibpur_files:
        print("Error: Shibpur tender files not found in workspace.")
        return

    print(f"Ingesting Tender Package ({len(shibpur_files)} files):")
    for f in shibpur_files:
        print(f"  - {os.path.basename(f)}")

    pipeline = TenderRAGPipeline()
    pipeline.clear()

    tender_id = pipeline.ingest_tender_set(shibpur_files)
    print(f"\nIndexed Tender Set ID: '{tender_id}'\n")

    output_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "top5_retrieval_scores.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    out_lines = []
    out_lines.append("=================================================================")
    out_lines.append("     BIDREADY TOP 5 RETRIEVED CHUNKS SCORE BREAKDOWN           ")
    out_lines.append("=================================================================\n")

    for idx, question in enumerate(QUESTION_SUITE, 1):
        print(f"[{idx}/70] Processing: {question[:45]}...", flush=True)
        top5_results = pipeline.search_engine.search(question, n_results=5, tender_id=tender_id)

        header_str = f"--- Q{idx}/{len(QUESTION_SUITE)}: {question} ---"
        out_lines.append(header_str)
        print(header_str)

        for rank, r in enumerate(top5_results, 1):
            chunk_text = r.chunk.text.replace("\n", " ").strip()[:200]
            info_str = (
                f"  Rank: #{rank} | Chunk ID: {r.chunk.chunk_id} | Source: {r.chunk.source_doc} (P{r.chunk.page_number}) | "
                f"Vector Score: {r.vector_score:.4f} | BM25 Score: {r.bm25_score:.4f} | "
                f"Reranker Score: {r.rerank_score:.4f} | Final Score: {r.combined_score:.4f}"
            )
            text_str = f"    Text Preview: {chunk_text}"
            
            out_lines.append(info_str)
            out_lines.append(text_str)
            print(info_str)
            print(text_str)

        out_lines.append("-" * 65 + "\n")
        print("-" * 65 + "\n")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

    print(f"\n[DONE] Diagnostic run complete! Report saved to: {output_file}")

if __name__ == "__main__":
    main()
