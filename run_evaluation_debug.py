import os
import sys

# Add current directory to path to resolve src imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.retriever import TenderRetriever

def format_explanation(cand: dict) -> str:
    reasons = []
    
    # 1. Check reranker score
    if cand["rerank_score"] > 0:
        reasons.append(f"Cross-Encoder Reranker confidence is positive ({cand['rerank_score']:.2f})")
    elif cand["rerank_score"] < 0:
        reasons.append(f"Cross-Encoder Reranker flagged negative attention ({cand['rerank_score']:.2f})")
        
    # 2. Check BM25 score
    if cand["norm_bm25"] > 0.7:
        reasons.append(f"Strong BM25 keyword matching (score: {cand['bm25_score']:.2f})")
    elif cand["norm_bm25"] > 0.3:
        reasons.append(f"Moderate BM25 keyword matching (score: {cand['bm25_score']:.2f})")
        
    # 3. Check Phrase Match
    if cand["phrase_boost"] > 0:
        reasons.append(f"Exact keyphrase match boost (+{cand['phrase_boost']:.1f})")
        
    # 4. Check Domain Boost
    if cand["domain_boost"] > 0:
        reasons.append(f"Domain rule intent boost (+{cand['domain_boost']:.1f})")
    elif cand["domain_boost"] < 0:
        reasons.append(f"Domain rule concept exclusion penalty ({cand['domain_boost']:.1f})")
        
    return "; ".join(reasons) if reasons else "Default fallback retrieval score"

def run_debug_loop():
    # Force stdout to UTF-8 to handle Unicode characters (like ₹)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("==================================================")
    print("      TENDER RETRIEVAL DEBUGGING CONSOLE          ")
    print("==================================================")
    
    retriever = TenderRetriever()
    
    print("\nInteractive Debugging Session Active. Enter query below. Type 'exit' to quit.\n")
    
    while True:
        try:
            query = input("\nEnter query: ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            break
            
        if not query:
            continue
        if query.lower() in ['exit', 'quit']:
            print("Exiting. Goodbye!")
            break
            
        print(f"\nAnalyzing retrieval steps for query: '{query}'")
        print("-" * 50)
        
        # Fetch debug information
        candidates = retriever.retrieve(query, debug=True)
        
        if not candidates:
            print("No matching candidate chunks found.")
            continue
            
        print(f"\nTop 3 Debug Candidates:")
        for idx, cand in enumerate(candidates[:3]):
            explanation = format_explanation(cand)
            
            print(f"\nCandidate {idx + 1} | Page {cand['metadata']['page_number']} | Final Score: {cand['score']:.4f}")
            print(f"  ├── Vector Similarity: {cand['sim']:.4f} (Raw Dist: {cand['dist']:.4f})")
            print(f"  ├── BM25 Score:        {cand['bm25_score']:.4f} (Normalized: {cand['norm_bm25']:.4f})")
            print(f"  ├── Phrase Boost:      {cand['phrase_boost']:.4f}")
            print(f"  ├── Domain Boost:      {cand['domain_boost']:.4f}")
            print(f"  ├── Rerank Score:      {cand['rerank_score']:.4f}")
            print(f"  └── Why Selected:      {explanation}")
            print(f"  [Text]: \"{cand['text'][:250]}...\"")
            
        print("\n" + "="*80)

if __name__ == "__main__":
    run_debug_loop()
