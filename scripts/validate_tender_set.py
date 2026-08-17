import os
import sys
import time
from typing import List, Dict, Any

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.pipeline import TenderRAGPipeline
from app.config.settings import TENDERS_DIR
from scripts.validate_retrieval import QUESTION_SUITE, open_native_file_dialog, scan_available_files

def run_tender_set_validation(files: List[str], mode_label: str = "MODE B: Multi-Document Tender Set"):
    print("\n=================================================================")
    print(f"        BIDREADY TENDER SET RETRIEVAL & CONFLICT AUDITOR        ")
    print("=================================================================")
    print(f"Mode: {mode_label}")
    print(f"Files Selected ({len(files)}):")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    print("=================================================================\n")

    pipeline = TenderRAGPipeline()
    pipeline.clear()

    # Ingest under unified tender set
    tender_id = pipeline.ingest_tender_set(files)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    summary_file = os.path.join(output_dir, "tender_set_validation_summary.md")
    report_file = os.path.join(output_dir, "tender_set_validation_report.txt")

    print(f"\n[EVAL] Running 70-Question Diagnostic Audit on Tender Set '{tender_id}'...")
    start_time = time.time()

    high_c = 0
    med_c = 0
    low_c = 0
    conflict_c = 0

    hits_top1 = 0
    hits_top3 = 0
    hits_top5 = 0
    hits_top10 = 0
    reciprocal_ranks = []

    md_rows = []
    report_lines = []
    report_lines.append("=================================================================")
    report_lines.append("     BIDREADY MULTI-DOCUMENT RETRIEVAL DIAGNOSTIC REPORT        ")
    report_lines.append("=================================================================")
    report_lines.append(f"Date/Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Tender ID: {tender_id}")
    report_lines.append(f"Indexed Documents ({len(files)}):")
    for af in files:
        report_lines.append(f"  - {os.path.basename(af)}")
    report_lines.append("=================================================================\n")

    for idx, question in enumerate(QUESTION_SUITE, 1):
        print(f"[{idx}/70] Auditing: {question[:45]}...", flush=True)
        top10_results = pipeline.search_engine.search(question, n_results=10, tender_id=tender_id)
        results = top10_results[:3]
        
        # Check conflict
        query_model = pipeline.search_engine.query_expander.expand_query(question)
        from app.conflict.conflict_detector import ConflictDetector
        conflict_report = ConflictDetector.detect_conflicts(query_model.requested_attribute, results)

        if conflict_report.has_conflict:
            conflict_c += 1
            conf = "⚠️ CONFLICT DETECTED"
        elif not results:
            conf = "🔴 UNSUPPORTED"
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

        # Determine evidence rank for benchmark metrics
        correct_rank = None
        for r_i, r in enumerate(top10_results, 1):
            if "🟢" in r.confidence or "🟡" in r.confidence:
                correct_rank = r_i
                break

        if correct_rank == 1:
            hits_top1 += 1
        if correct_rank and correct_rank <= 3:
            hits_top3 += 1
        if correct_rank and correct_rank <= 5:
            hits_top5 += 1
        if correct_rank and correct_rank <= 10:
            hits_top10 += 1
        
        if correct_rank:
            reciprocal_ranks.append(1.0 / correct_rank)
        else:
            reciprocal_ranks.append(0.0)

        if not results:
            snip = "No evidence found."
            cite = "N/A"
        else:
            top_r = results[0]
            snip = top_r.chunk.text.replace("\n", " ")[:140] + "..."
            cite = f"{top_r.chunk.source_doc} ({top_r.chunk.document_type}, P{top_r.chunk.page_number})"

        safe_snip = snip.replace("|", "/")
        md_rows.append(f"| Q{idx} | {question} | {conf} | {safe_snip} | {cite} |")

        report_lines.append(f"--- Q{idx}/{len(QUESTION_SUITE)}: {question} ---")
        report_lines.append(f"Query Category: {query_model.query_category} | Target: {query_model.target_type} ({query_model.requested_attribute})")
        report_lines.append(f"Confidence: {conf} | First Answerable Evidence Rank: {correct_rank if correct_rank else 'NOT FOUND IN TOP 10'}")
        if conflict_report.has_conflict:
            report_lines.append(f"Conflict Status: {conflict_report.resolution_summary}")

        report_lines.append("  [TOP 10 CANDIDATES SCORE BREAKDOWN]")
        for r_i, r in enumerate(top10_results, 1):
            report_lines.append(
                f"    #{r_i} [{r.confidence}] Doc: {r.chunk.source_doc} (P{r.chunk.page_number}) | "
                f"Vector: {r.vector_score:.3f} | BM25: {r.bm25_score:.3f} | Exact: {r.exact_match_score:.3f} | "
                f"Rerank: {r.rerank_score:.3f} | Final: {r.combined_score:.4f}"
            )
            report_lines.append(f"       Text: {r.chunk.text.strip()[:160]}...")
        report_lines.append("-" * 65 + "\n")

    elapsed = time.time() - start_time
    total_q = len(QUESTION_SUITE)
    rec1 = (hits_top1 / total_q) * 100
    rec3 = (hits_top3 / total_q) * 100
    rec5 = (hits_top5 / total_q) * 100
    rec10 = (hits_top10 / total_q) * 100
    mrr = sum(reciprocal_ranks) / total_q if total_q > 0 else 0.0

    md_header = []
    md_header.append("# Multi-Document Tender Set Validation Report\n")
    md_header.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    md_header.append(f"**Tender ID:** `{tender_id}`  ")
    md_header.append(f"**Documents Audited ({len(files)}):** {', '.join([os.path.basename(f) for f in files])}  \n")
    md_header.append("### 🎯 Retrieval Benchmark Metrics")
    md_header.append(f"- **Recall@1:** `{rec1:.1f}%` ({hits_top1}/{total_q})")
    md_header.append(f"- **Recall@3:** `{rec3:.1f}%` ({hits_top3}/{total_q})")
    md_header.append(f"- **Recall@5:** `{rec5:.1f}%` ({hits_top5}/{total_q})")
    md_header.append(f"- **Recall@10:** `{rec10:.1f}%` ({hits_top10}/{total_q})")
    md_header.append(f"- **Mean Reciprocal Rank (MRR):** `{mrr:.4f}`  \n")
    md_header.append(f"**Confidence Summary:** 🟢 High: {high_c} | 🟡 Medium: {med_c} | ⚠️ Conflicts: {conflict_c} | 🔴 Low/Unsupported: {low_c}  \n")
    md_header.append("## 70-Question Multi-Document Retrieval & Conflict Matrix\n")
    md_header.append("| Q# | Audit Question | Confidence / Conflict | Top Evidence Snippet | Source Citation & Document Type |")
    md_header.append("|---|---|---|---|---|")

    full_md = "\n".join(md_header + md_rows)
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(full_md)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n[DONE] Diagnostic Validation finished in {elapsed:.1f}s!")
    print(f"Recall@1: {rec1:.1f}% | Recall@3: {rec3:.1f}% | Recall@5: {rec5:.1f}% | Recall@10: {rec10:.1f}% | MRR: {mrr:.4f}")
    print(f"High: {high_c} | Medium: {med_c} | Conflicts: {conflict_c} | Low: {low_c}")
    print(f"Summary written to: {summary_file}")
    print(f"Detailed report written to: {report_file}")

def main():
    available_files = scan_available_files()
    print("=================================================================")
    print("      BIDREADY MULTI-DOCUMENT TENDER SET VALIDATOR              ")
    print("=================================================================")
    print("Select Evaluation Mode:")
    print("  [1] MODE A: Single Document (Pure Retrieval Test)")
    print("  [2] MODE B: Multi-Document Tender Set (NIT + BOQ + Corrigendum)")
    print("  [0] 📁 Browse via Windows File Explorer")

    choice = input("\nEnter choice (0, 1, or 2): ").strip()

    if choice == "0":
        files = open_native_file_dialog()
        if files:
            mode_lbl = "MODE A" if len(files) == 1 else "MODE B: Multi-Document Tender Set"
            run_tender_set_validation(files, mode_label=mode_lbl)
            return
        choice = "2"

    if choice == "1":
        print("\nSelect a single file from workspace:")
        for idx, fp in enumerate(available_files, 1):
            print(f"  [{idx}] {os.path.relpath(fp, TENDERS_DIR)}")
        pick = input("Enter file number: ").strip()
        try:
            sel_file = available_files[int(pick) - 1]
            run_tender_set_validation([sel_file], mode_label="MODE A: Single Document")
        except Exception:
            print("Invalid selection. Exiting.")
    else:
        print("\nOpening File Explorer to select multi-document tender package...")
        files = open_native_file_dialog()
        if not files:
            print("Defaulting to Shibpur tender package...")
            files = [f for f in available_files if "shibpur" in f.lower()]
        if files:
            run_tender_set_validation(files, mode_label="MODE B: Multi-Document Tender Set")
        else:
            print("No files selected. Exiting.")

if __name__ == "__main__":
    main()
