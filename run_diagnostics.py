import sys
import os

# Add current directory to path to resolve src imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.retriever import TenderRetriever

FAILED_QUESTIONS = [
    "Where is the work located?",
    "Which government department does the tender belong to?",
    "What type of tender is this?",
    "Under what condition is the Integrity Pact required?"
]

def format_explanation(cand: dict) -> str:
    reasons = []
    if cand["rerank_score"] != 0.0:
        reasons.append(f"Rerank: {cand['rerank_score']:.2f}")
    if cand["phrase_boost"] != 0.0:
        reasons.append(f"Phrase: +{cand['phrase_boost']:.1f}")
    if cand["domain_boost"] != 0.0:
        reasons.append(f"Domain: {cand['domain_boost']:.1f}")
    reasons.append(f"BM25: {cand['bm25_score']:.2f} (norm: {cand['norm_bm25']:.2f})")
    reasons.append(f"VecSim: {cand['sim']:.2f}")
    return "; ".join(reasons)

def run_diagnostics():
    # Force stdout to UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    retriever = TenderRetriever()
    
    for query in FAILED_QUESTIONS:
        print(f"\n======================================================================")
        print(f"QUERY: {query}")
        print(f"======================================================================")
        
        candidates = retriever.retrieve(query, debug=True)
        
        for idx, cand in enumerate(candidates[:3]):
            print(f"\nResult #{idx+1} | Page {cand['metadata']['page_number']} | Final Score: {cand['score']:.4f}")
            print(f"Details: {format_explanation(cand)}")
            print(f"Text: \"{cand['text'][:400]}...\"")
            print("-" * 70)

if __name__ == "__main__":
    run_diagnostics()
