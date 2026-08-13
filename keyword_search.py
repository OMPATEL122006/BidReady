import os
import sys

# Add current directory to path to resolve src imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import TENDERS_DIR
from src.document_parser import extract_text_from_pdf

def search_keywords_in_pdf():
    # Force stdout to UTF-8 to handle Unicode characters (like ₹)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    pdf_path = os.path.join(TENDERS_DIR, "tender.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find tender.pdf in Tenders folder: {pdf_path}")
        sys.exit(1)
        
    print(f"--- 1. Extracting text from: {pdf_path} ---")
    extracted_text = extract_text_from_pdf(pdf_path)
    print(f"Loaded {len(extracted_text)} pages.\n")
    
    keywords = ["turnover", "iso", "certificat"]
    
    print("--- 2. Running Case-Insensitive Keyword Search ---")
    for keyword in keywords:
        print(f"\n==================================================")
        print(f"SEARCH KEYWORD: \"{keyword}\"")
        print(f"==================================================")
        
        matches_found = 0
        
        for page_num, text in extracted_text.items():
            # Case-insensitive search
            lower_text = text.lower()
            start_idx = 0
            
            while True:
                idx = lower_text.find(keyword, start_idx)
                if idx == -1:
                    break
                    
                matches_found += 1
                
                # Determine print window boundaries (~300 characters total around the match)
                window_start = max(0, idx - 150)
                window_end = min(len(text), idx + len(keyword) + 150)
                
                snippet = text[window_start:window_end]
                
                # Formatting snippet output
                print(f"\n[Match #{matches_found}] Page {page_num} | Char Position {idx}")
                print(f"--------------------------------------------------")
                print(f"...{snippet}...")
                print(f"--------------------------------------------------")
                
                # Move forward past this match to find next occurrences on same page
                start_idx = idx + len(keyword)
                
        if matches_found == 0:
            print(f"No occurrences of \"{keyword}\" were found in the document.")

if __name__ == "__main__":
    search_keywords_in_pdf()
