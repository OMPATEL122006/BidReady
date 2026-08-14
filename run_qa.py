import os
import sys
import argparse

# Add current directory to path to resolve src imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.document_parser import TenderParser
from src.chunker import chunk_document
from src.database import TenderDatabase
from src.retriever import TenderRetriever
from src.generator import TenderAnswerGenerator

def main():
    # Force stdout to UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description="BidReady Tender QA System")
    parser.add_argument("--pdf", type=str, help="Path to a tender PDF to index before asking questions")
    args = parser.parse_args()

    # 1. Index PDF if specified
    if args.pdf:
        if not os.path.exists(args.pdf):
            print(f"Error: Specified PDF file does not exist: {args.pdf}")
            return
            
        print(f"\n--- Parsing and Indexing {os.path.basename(args.pdf)} ---")
        pdf_parser = TenderParser(args.pdf)
        pages = pdf_parser.parse()
        print(f"Parsed {len(pages)} pages.")
        
        chunks = chunk_document(pages)
        print(f"Generated {len(chunks)} chunks.")
        
        db = TenderDatabase()
        db.clear_database()
        db.add_chunks(chunks)
        print("Indexed chunks in Chroma database successfully.")

    # 2. Initialize Retriever and Generator
    print("\nInitializing Tender Retriever and Groq Answer Generator...")
    try:
        retriever = TenderRetriever()
        generator = TenderAnswerGenerator(retriever)
        print("BidReady QA Engine initialized successfully! Type 'exit' or 'quit' to close.")
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    # 3. Interactive QA loop
    print("=================================================================")
    print("                    BIDREADY TENDER QA TERMINAL                  ")
    print("=================================================================")
    
    while True:
        try:
            query = input("\nAsk a question about the indexed tender: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
                
            print("\nSearching and generating answer...")
            result = generator.generate_answer(query)
            
            print("\n--- Answer ---")
            print(result["answer"])
            print("--------------")
            print(f"Confidence Label: {result['confidence']}")
            
            if result["sources"]:
                print("\nSources referenced:")
                for idx, src in enumerate(result["sources"]):
                    snippet = src["text"][:120].replace("\n", " ") + "..."
                    print(f"  [{idx + 1}] Page {src['page']} ({src['confidence']}) - Snippet: {snippet}")
                    
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
