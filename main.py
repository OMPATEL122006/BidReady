import sys
import os
from app.pipeline import TenderRAGPipeline
from app.config.settings import TENDERS_DIR

def main():
    print("=================================================================")
    print("           BIDREADY GOVERNMENT TENDER QA TERMINAL                ")
    print("=================================================================")

    pipeline = TenderRAGPipeline()
    
    # Automatically ingest any tender files in TENDERS_DIR if present
    if os.path.exists(TENDERS_DIR):
        tender_files = [os.path.join(TENDERS_DIR, f) for f in os.listdir(TENDERS_DIR) if f.lower().endswith((".pdf", ".xlsx", ".xls"))]
        if tender_files:
            print(f"Indexing {len(tender_files)} tender document(s) from {TENDERS_DIR}...")
            for tf in tender_files:
                pipeline.ingest_file(tf)
            print("Tender indexing complete!\n")

    print("Type your questions below. Type 'exit' or 'quit' to close.\n")
    while True:
        try:
            query = input("Ask Tender Question > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Closing BidReady terminal. Goodbye!")
                break

            print("\nSearching and generating answer...")
            res = pipeline.ask(query)
            print("\n--- Answer ---")
            print(res["answer"])
            print("\nConfidence:", res["confidence"])
            print("Sources:")
            for s in res["sources"]:
                print(f"  - Page {s['page']} ({s['confidence']})")
            print("-" * 65 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nClosing terminal. Goodbye!")
            break

if __name__ == "__main__":
    main()
