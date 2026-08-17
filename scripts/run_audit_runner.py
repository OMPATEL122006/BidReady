import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.validate_retrieval import QUESTION_SUITE, scan_available_files
from app.pipeline import TenderRAGPipeline

def run_fast_audit(target_keyword="Shibpur"):
    all_files = scan_available_files()
    selected_files = [f for f in all_files if target_keyword.lower() in f.lower()]
    if not selected_files:
        selected_files = all_files[:2]

    print(f"Indexing {len(selected_files)} document(s) for keyword '{target_keyword}'...")
    pipeline = TenderRAGPipeline()
    pipeline.clear()

    for fp in selected_files:
        pipeline.ingest_file(fp)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    summary_file = os.path.join(output_dir, "retrieval_validation_summary.md")
    report_file = os.path.join(output_dir, "retrieval_validation_report.txt")

    md_lines = []
    md_lines.append("# Retrieval & Context Augmentation Audit Report\n")
    md_lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    md_lines.append(f"**Documents Audited ({len(selected_files)}):** " + ", ".join([os.path.basename(f) for f in selected_files]) + "  \n")
    md_lines.append("## 70-Question Retrieval Performance Matrix\n")
    md_lines.append("| Q# | Audit Question | Retrieval Confidence | Top Augmented Evidence Snippet | Source Citation |")
    md_lines.append("|---|---|---|---|---|")

    report_lines = []
    report_lines.append("=================================================================")
    report_lines.append("     BIDREADY RETRIEVAL & AUGMENTATION VALIDATION REPORT         ")
    report_lines.append("=================================================================")

    high_c = 0
    med_c = 0
    low_c = 0

    start_time = time.time()
    for idx, question in enumerate(QUESTION_SUITE, 1):
        results = pipeline.search_engine.search(question, n_results=3)
        if not results:
            conf = "🔴 UNSUPPORTED"
            snip = "No evidence found in uploaded document."
            cite = "N/A"
            low_c += 1
        else:
            top_r = results[0]
            conf = top_r.confidence
            if "🟢" in conf:
                high_c += 1
            elif "🟡" in conf:
                med_c += 1
            else:
                low_c += 1

            snip = top_r.chunk.text.replace("\n", " ")[:150] + "..."
            cite = f"Page {top_r.chunk.page_number} ({top_r.chunk.source_doc})"

        safe_snip = snip.replace("|", "/")
        md_lines.append(f"| Q{idx} | {question} | {conf} | {safe_snip} | {cite} |")

        report_lines.append(f"--- Q{idx}: {question} ---")
        report_lines.append(f"Confidence: {conf}")
        if results:
            for r_i, r in enumerate(results, 1):
                report_lines.append(f"  Chunk #{r_i} [Page {r.chunk.page_number}]: {r.chunk.text.strip()}")
        report_lines.append("-" * 65 + "\n")

    elapsed = time.time() - start_time
    md_lines.insert(4, f"**Total Questions:** 70 | **High Confidence:** {high_c} | **Medium Confidence:** {med_c} | **Low/Unsupported:** {low_c}\n")

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Audit completed in {elapsed:.1f}s!")
    print(f"High: {high_c} | Med: {med_c} | Low: {low_c}")
    print(f"Summary written to: {summary_file}")

if __name__ == "__main__":
    kw = sys.argv[1] if len(sys.argv) > 1 else "Shibpur"
    run_fast_audit(kw)
