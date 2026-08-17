import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.pipeline import TenderRAGPipeline

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description="BidReady Tender QA System")
    parser.add_argument("--pdf", type=str, help="Path to a tender PDF to index")
    parser.add_argument("--xlsx", type=str, help="Path to a BOQ Excel file to index")
    parser.add_argument("--dir", type=str, help="Path to a directory containing files to index")
    args = parser.parse_args()

    pipeline = TenderRAGPipeline()
    pipeline.clear()

    files_to_index = []
    if args.pdf:
        files_to_index.append(args.pdf)
    if args.xlsx:
        files_to_index.append(args.xlsx)
    if args.dir and os.path.exists(args.dir):
        for root, _, files in os.walk(args.dir):
            for file in files:
                if file.lower().endswith((".pdf", ".xlsx", ".xls")):
                    files_to_index.append(os.path.join(root, file))

    if not files_to_index:
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tenders")
        if os.path.exists(default_dir):
            for root, _, files in os.walk(default_dir):
                for file in files:
                    if file.lower().endswith((".pdf", ".xlsx", ".xls")):
                        files_to_index.append(os.path.join(root, file))

    print(f"\nIndexing {len(files_to_index)} file(s)...")
    for fp in files_to_index:
        pipeline.ingest_file(fp)

    print("\nBidReady QA Engine initialized successfully! Type 'exit' or 'quit' to close.")
    print("=================================================================")
    print("                    BIDREADY TENDER QA TERMINAL                  ")
    print("=================================================================\n")

    while True:
        try:
            query = input("\nAsk Question about Tender > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting. Goodbye!")
                break

            print("\nSearching and generating answer...")
            res = pipeline.ask(query)
            print("\n--- Answer ---")
            print(res["answer"])
            print("\nConfidence:", res["confidence"])
            print("Sources:")
            for s in res["sources"]:
                print(f"  - Page {s['page']} ({s['confidence']})")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

if __name__ == "__main__":
    main()
