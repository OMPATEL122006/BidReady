import os

from app.pipeline import TenderRAGPipeline
from app.config.settings import TENDERS_DIR


def main():

    print("=" * 65)
    print(" BIDREADY GOVERNMENT TENDER QA TERMINAL ")
    print("=" * 65)

    pipeline = TenderRAGPipeline()

    if not os.path.exists(TENDERS_DIR):
        print(f"Tender directory not found: {TENDERS_DIR}")
        return

    files = [
        os.path.join(TENDERS_DIR, f)
        for f in os.listdir(TENDERS_DIR)
        if f.lower().endswith((".pdf", ".xlsx", ".xls"))
    ]

    if not files:
        print("No tender documents found.")
        return

    # ONE PDF + its BOQ/XLS = ONE TENDER SET
    tender_id = pipeline.ingest_tender_set(
        files,
        tender_id=None,
    )

    print(f"\nTender Set ID: {tender_id}")
    print(f"Indexed {len(files)} document(s).")
    print("Tender indexing complete!\n")

    print("Type your questions below.")
    print("Type 'exit' or 'quit' to close.\n")

    while True:

        try:
            query = input(
                "Ask Tender Question > "
            ).strip()

            if not query:
                continue

            if query.lower() in {
                "exit",
                "quit",
                "q",
            }:
                print(
                    "\nClosing BidReady terminal. Goodbye!"
                )
                break

            print("\nSearching...\n")

            result = pipeline.ask(
                query,
                n_results=5,
                tender_id=tender_id,
            )

            print("--- Answer ---")
            print(result["answer"])

            print(
                "\nConfidence:",
                result["confidence"],
            )

            print("\nSources:")

            for source in result.get("sources", []):
                print(
                    f" - {source.get('doc', '')} | "
                    f"Page {source.get('page', '')} | "
                    f"{source.get('confidence', '')}"
                )

            print("-" * 65)

        except (KeyboardInterrupt, EOFError):
            print(
                "\nClosing terminal. Goodbye!"
            )
            break


if __name__ == "__main__":
    main()