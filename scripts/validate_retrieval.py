import os
import sys
import time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.pipeline import TenderRAGPipeline
from app.config.settings import TENDERS_DIR

# Comprehensive 70-Question Government Tender Retrieval Audit Suite
QUESTION_SUITE = [
    # --- Category 1: Work Name & NIT Identification (1-5) ---
    "What is the name or title of the work?",
    "What is the tender enquiry number or NIT reference number?",
    "Which institute, department, or organization issued this tender?",
    "What is the location or site of the proposed work?",
    "What is the tender type or mode of bidding?",

    # --- Category 2: Financial Estimates, EMD & Bid Security (6-15) ---
    "What is the estimated cost of the tender?",
    "What is the EMD (Earnest Money Deposit) amount or percentage?",
    "What are the acceptable modes or forms of EMD payment?",
    "To whom should the EMD demand draft be drawn in favor of?",
    "Where and by when must the hard copy of EMD be submitted?",
    "Is Micro and Small Enterprises (MSE) or MSME EMD exemption allowed?",
    "What is the required Performance Security or Performance Guarantee percentage?",
    "What is the Security Deposit percentage to be deducted from bills?",
    "What is the validity period of the performance security?",
    "Is tender document fee or cost applicable?",

    # --- Category 3: Dates, Deadlines & Schedules (16-25) ---
    "What is the last date and time for online bid submission?",
    "When and at what time will the technical/eligibility bid be opened?",
    "When will the financial bid be opened?",
    "What is the bid validity period in days?",
    "What is the period of completion or execution time allowed for the work?",
    "When is the pre-bid meeting scheduled, if any?",
    "What is the date of publication or issue of the tender notice?",
    "What is the deadline for seeking clarifications or queries?",
    "What is the work start date after issuance of Letter of Intent/Work Order?",
    "What is the schedule of critical dates?",

    # --- Category 4: Eligibility & Technical Criteria (26-35) ---
    "What is the required experience for similar completed works in the last 7 years?",
    "What is the cost threshold for 3 similar completed works?",
    "What is the cost threshold for 2 similar completed works?",
    "What is the cost threshold for 1 similar completed work?",
    "What is the average annual financial turnover requirement?",
    "Is Solvency Certificate required, and for what value?",
    "Is Joint Venture (JV) or Consortium allowed in this tender?",
    "Is subcontracting of work allowed?",
    "Is valid Trade License or Enlistment Certificate required?",
    "What class or category of contractor enlistment is required?",

    # --- Category 5: Statutory, Tax & Compliance Documents (36-45) ---
    "Is GST registration certificate required?",
    "Is Permanent Account Number (PAN) card required?",
    "Is Employees' Provident Fund (EPF) registration required?",
    "Is Employees' State Insurance (ESI) registration required?",
    "Is Income Tax Return (ITR) for past financial years required?",
    "Is Certificate of Non-Blacklisting or Affidavit required?",
    "Is Site Inspection Certificate or Undertaking required?",
    "What documents must be uploaded in Cover 1 (Technical Bid)?",
    "What documents must be uploaded in Cover 2 (Financial Bid)?",
    "Is valid Electrical License required for electrical works?",

    # --- Category 6: Maintenance, Defect Liability & Guarantees (46-55) ---
    "What is the Period of Maintenance or Defect Liability Period (DLP)?",
    "Who is responsible for repair of defects during the maintenance period?",
    "What is the warranty or guarantee requirement for supplied materials?",
    "What happens if defective work is not rectified within the specified time?",
    "Is security deposit refunded after completion of the defect liability period?",
    "Is insurance coverage required for workers or site equipment?",
    "What safety measures and compliance are mandatory on site?",
    "What testing or quality control reports are required before billing?",
    "Is final clearance or completion certificate required for payment?",
    "What happens if the contractor fails to start the work on time?",

    # --- Category 7: Payment Terms, Price Escalation & Advances (56-65) ---
    "Is mobilization advance or any advance payment allowed?",
    "What are the terms of running account (RA) bill payments?",
    "Is price escalation or price variation clause allowed?",
    "What taxes, duties, or cesses are included or excluded in quoted rates?",
    "Is Labour Welfare Cess applicable, and at what percentage?",
    "What percentage of payment is retained until final completion?",
    "What is the mode of bill payment (e.g. electronic transfer/e-payment)?",
    "Are statutory deductions made at source from contractor bills?",
    "What is the time limit for processing running bills?",
    "What happens in case of delay in work completion (Liquidated Damages)?",

    # --- Category 8: Contact Details, Site Visit & Queries (66-70) ---
    "Who is the contact person for site inspection or queries?",
    "What is the phone or mobile number of the contact person?",
    "What is the official email address for sending tender queries?",
    "Where must the physical site visit be conducted?",
    "Where must the online tender bid be uploaded (portal URL)?"
]

def scan_available_files() -> List[str]:
    files = []
    if os.path.exists(TENDERS_DIR):
        for root, _, f_names in os.walk(TENDERS_DIR):
            for fn in f_names:
                if fn.lower().endswith((".pdf", ".xlsx", ".xls")):
                    files.append(os.path.join(root, fn))
    return files

def open_native_file_dialog() -> List[str]:
    """
    Opens standard Windows native File Explorer dialog to browse and pick tender files visually.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_paths = filedialog.askopenfilenames(
            title="Select Tender PDF or BOQ Excel Document(s)",
            initialdir=TENDERS_DIR if os.path.exists(TENDERS_DIR) else os.getcwd(),
            filetypes=[
                ("Tender Documents (*.pdf, *.xlsx, *.xls)", "*.pdf;*.xlsx;*.xls"),
                ("PDF Files (*.pdf)", "*.pdf"),
                ("Excel Files (*.xlsx, *.xls)", "*.xlsx;*.xls"),
                ("All Files (*.*)", "*.*")
            ]
        )
        root.destroy()
        if selected_paths:
            return list(selected_paths)
    except Exception as e:
        print(f"Native file dialog note: {e}")
    return []

def prompt_file_selection(available_files: List[str]) -> List[str]:
    print("\n=================================================================")
    print("        BIDREADY RETRIEVAL & AUGMENTATION VALIDATOR              ")
    print("=================================================================")

    # Attempt native File Explorer GUI dialog first
    print("\nOpening Windows File Explorer dialog window to select files...")
    gui_selected = open_native_file_dialog()
    if gui_selected:
        print(f"\n✅ Selected {len(gui_selected)} file(s) via File Explorer:")
        for gf in gui_selected:
            print(f"  - {os.path.basename(gf)} ({gf})")
        return gui_selected

    print("\nFile Explorer dialog closed or canceled. Falling back to menu selection:")

    if not available_files:
        print(f"No tender files found in '{TENDERS_DIR}'.")
        custom = input("Enter path to a PDF/Excel file or directory to validate: ").strip()
        if os.path.isdir(custom):
            return [os.path.join(custom, f) for f in os.listdir(custom) if f.lower().endswith((".pdf", ".xlsx", ".xls"))]
        elif os.path.isfile(custom):
            return [custom]
        else:
            print("Invalid path provided. Exiting.")
            sys.exit(1)

    print("\nAvailable Tender Documents in Workspace:")
    print("  [0] 📁 Re-open Windows File Explorer browser dialog")
    for idx, fp in enumerate(available_files, 1):
        rel_path = os.path.relpath(fp, TENDERS_DIR)
        size_kb = os.path.getsize(fp) / 1024.0
        print(f"  [{idx}] {rel_path} ({size_kb:.1f} KB)")

    print(f"  [{len(available_files) + 1}] Select ALL files")
    print(f"  [{len(available_files) + 2}] Specify a custom file/directory path")

    choice = input("\nSelect document option (0 to browse window, or numbers e.g. 1,2): ").strip()
    
    if choice == "0":
        gui_selected = open_native_file_dialog()
        if gui_selected:
            return gui_selected
        return available_files
    elif choice == str(len(available_files) + 1) or choice.lower() in ["all", "a"]:
        return available_files
    elif choice == str(len(available_files) + 2):
        custom = input("Enter custom path: ").strip()
        if os.path.isdir(custom):
            return [os.path.join(custom, f) for f in os.listdir(custom) if f.lower().endswith((".pdf", ".xlsx", ".xls"))]
        elif os.path.isfile(custom):
            return [custom]
        else:
            print("Invalid path. Exiting.")
            sys.exit(1)
    else:
        selected = []
        try:
            parts = [p.strip() for p in choice.split(",") if p.strip()]
            for p in parts:
                num = int(p)
                if 1 <= num <= len(available_files):
                    selected.append(available_files[num - 1])
        except Exception:
            pass
        if not selected:
            print("Defaulting to selecting ALL available files.")
            return available_files
        return selected

def run_retrieval_validation():
    available_files = scan_available_files()
    selected_files = prompt_file_selection(available_files)

    print(f"\n[INIT] Initializing Tender RAG Pipeline (Retrieval + Augmentation ONLY)...")
    pipeline = TenderRAGPipeline()
    pipeline.clear()

    print(f"[INGEST] Ingesting {len(selected_files)} document(s)...")
    for fp in selected_files:
        pipeline.ingest_file(fp)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    report_file_path = os.path.join(output_dir, "retrieval_validation_report.txt")
    summary_file_path = os.path.join(output_dir, "retrieval_validation_summary.md")

    print(f"\n[EVAL] Running 70-Question Retrieval & Context Augmentation Audit...")
    start_time = time.time()

    report_lines = []
    report_lines.append("=================================================================")
    report_lines.append("     BIDREADY RETRIEVAL & AUGMENTATION VALIDATION REPORT         ")
    report_lines.append("=================================================================")
    report_lines.append(f"Date/Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Indexed Documents ({len(selected_files)}):")
    for sf in selected_files:
        report_lines.append(f"  - {os.path.basename(sf)}")
    report_lines.append("=================================================================\n")

    summary_rows = []
    high_count = 0
    med_count = 0
    low_count = 0

    for idx, question in enumerate(QUESTION_SUITE, 1):
        print(f"[{idx}/{len(QUESTION_SUITE)}] Retrieving evidence for: '{question}'...")
        
        # Perform Hybrid Search + Reranking + Context Expansion + Validation ONLY (No LLM)
        results = pipeline.search_engine.search(question, n_results=3)

        report_lines.append(f"--- Q{idx}/{len(QUESTION_SUITE)}: {question} ---")
        if not results:
            report_lines.append("Confidence: 🔴 UNSUPPORTED / NO EVIDENCE FOUND")
            report_lines.append("Retrieved Evidence: None\n")
            summary_rows.append((idx, question, "🔴 UNSUPPORTED", "No evidence retrieved", "N/A"))
            low_count += 1
            continue

        top_res = results[0]
        conf = top_res.confidence
        if "🟢" in conf:
            high_count += 1
        elif "🟡" in conf:
            med_count += 1
        else:
            low_count += 1

        report_lines.append(f"Overall Confidence: {conf}")
        report_lines.append("Retrieved Augmented Evidence Chunks (Top 3):")

        top_page = top_res.chunk.page_number
        top_snippet = top_res.chunk.text.replace("\n", " ")[:160] + "..."

        summary_rows.append((idx, question, conf, top_snippet, f"Page {top_page}"))

        for r_idx, res in enumerate(results, 1):
            c = res.chunk
            report_lines.append(f"  [Chunk #{r_idx}] Source: {c.source_doc} | Page: {c.page_number} | Type: {c.chunk_type} | Confidence: {res.confidence}")
            report_lines.append(f"  Score: Combined={res.combined_score:.4f} | Rerank={res.rerank_score:.4f} | BM25={res.bm25_score:.4f} | Exact={res.exact_match_score:.4f}")
            report_lines.append("  Augmented Text Context:")
            for line in c.text.splitlines():
                report_lines.append(f"    {line}")
            report_lines.append("")

        report_lines.append("-" * 65 + "\n")

    elapsed_sec = time.time() - start_time
    report_lines.append("=================================================================")
    report_lines.append("                    VALIDATION SUMMARY STATISTICS                ")
    report_lines.append("=================================================================")
    report_lines.append(f"Total Questions Audited: {len(QUESTION_SUITE)}")
    report_lines.append(f"🟢 High Confidence (Exact Match): {high_count} ({high_count/len(QUESTION_SUITE)*100:.1f}%)")
    report_lines.append(f"🟡 Medium Confidence (Supporting Evidence): {med_count} ({med_count/len(QUESTION_SUITE)*100:.1f}%)")
    report_lines.append(f"🔴 Low / Unsupported Confidence: {low_count} ({low_count/len(QUESTION_SUITE)*100:.1f}%)")
    report_lines.append(f"Total Evaluation Time: {elapsed_sec:.2f} seconds")
    report_lines.append("=================================================================")

    # Write detailed text report
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # Write clean Markdown summary report
    md_lines = []
    md_lines.append("# Retrieval & Context Augmentation Audit Report\n")
    md_lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    md_lines.append(f"**Documents Audited ({len(selected_files)}):** " + ", ".join([os.path.basename(f) for f in selected_files]) + "  ")
    md_lines.append(f"**Total Questions:** {len(QUESTION_SUITE)} | **High Confidence:** {high_count} | **Medium Confidence:** {med_count} | **Low/Unsupported:** {low_count}\n")
    md_lines.append("## Retrieval Performance Matrix\n")
    md_lines.append("| Q# | Audit Question | Retrieval Confidence | Top Augmented Evidence Snippet | Source Citation |")
    md_lines.append("|---|---|---|---|---|")
    for q_idx, q_text, q_conf, q_snip, q_cite in summary_rows:
        safe_snip = q_snip.replace("|", "/")
        md_lines.append(f"| Q{q_idx} | {q_text} | {q_conf} | {safe_snip} | {q_cite} |")

    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n=================================================================")
    print(f"✅ RETRIEVAL VALIDATION COMPLETE ({elapsed_sec:.1f}s)!")
    print(f"🟢 High Confidence: {high_count}/{len(QUESTION_SUITE)} | 🟡 Medium: {med_count}/{len(QUESTION_SUITE)} | 🔴 Low: {low_count}/{len(QUESTION_SUITE)}")
    print(f"Full Evidence Report: {report_file_path}")
    print(f"Summary Table:        {summary_file_path}")
    print(f"=================================================================\n")

if __name__ == "__main__":
    run_retrieval_validation()
