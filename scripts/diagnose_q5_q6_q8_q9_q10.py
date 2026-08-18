import os
import sys
import json
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.pipeline import TenderRAGPipeline
from scripts.validate_retrieval import scan_available_files

DIAGNOSTIC_QUESTIONS = [
    (5, "What is the tender type or mode of bidding?"),
    (6, "What is the estimated cost of the tender?"),
    (8, "What are the acceptable modes or forms of EMD payment?"),
    (9, "To whom should the EMD demand draft be drawn in favor of?"),
    (10, "Where and by when must the hard copy of EMD be submitted?")
]

def main():
    print("=================================================================")
    print("   BIDREADY DECOUPLED ANSWERABILITY VALIDATOR DIAGNOSTIC TEST   ")
    print("   Target Questions: Q5, Q6, Q8, Q9, Q10                        ")
    print("=================================================================\n")

    available_files = scan_available_files()
    shibpur_files = [f for f in available_files if "shibpur" in f.lower()]

    if not shibpur_files:
        print("Error: Shibpur tender files not found in workspace.")
        return

    pipeline = TenderRAGPipeline()
    pipeline.clear()

    print(f"[INGEST] Ingesting Shibpur Tender Set ({len(shibpur_files)} files)...")
    tender_id = pipeline.ingest_tender_set(shibpur_files)
    print(f"[INGEST] Tender Set ingested under ID: {tender_id}\n")

    for q_idx, question in DIAGNOSTIC_QUESTIONS:
        print("=" * 80)
        print(f"DIAGNOSTIC Q{q_idx}: {question}")
        print("=" * 80)

        # 1. Fetch Top 5 Candidates after Reranker
        top_results = pipeline.search_engine.search(question, n_results=5, tender_id=tender_id)

        print("\n--- RETRIEVED TOP CANDIDATES EVIDENCE ---")
        for r_i, r in enumerate(top_results, 1):
            print(f"  [Candidate #{r_i}] Doc: {r.chunk.source_doc} (P{r.chunk.page_number}) | Combined Score: {r.combined_score:.4f}")
            print(f"    Text: {r.chunk.text.strip()}")
            print("-" * 65)

        # 2. Standalone Answerability Validator Decision
        val_report = pipeline.search_engine.validator.validate_answerability(question, top_results)
        ans_status = val_report["answerable"]

        if ans_status == "YES":
            final_conf = "🟢 HIGH CONFIDENCE"
        elif ans_status == "PARTIAL":
            final_conf = "🟡 MEDIUM CONFIDENCE"
        else:
            final_conf = "🔴 LOW CONFIDENCE"

        print("\n--- STRUCTURED ANSWERABILITY VALIDATOR OUTPUT ---")
        print(json.dumps(val_report, indent=2))
        print(f"\nFINAL CLASSIFICATION: {final_conf}\n")

if __name__ == "__main__":
    main()
