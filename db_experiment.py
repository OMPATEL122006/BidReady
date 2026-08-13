import os
import sys

# Add current directory to path to resolve src imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import TENDERS_DIR, STOP_WORDS
from src.document_parser import extract_text_from_pdf
from src.chunker import create_chunks
from src.database import get_chroma_client, get_or_create_collection, store_chunks_in_chroma, query_chroma

def compute_keyword_score(query: str, doc_text: str) -> float:
    """
    Computes a simple Jaccard-like overlap score between query keywords and document words.
    """
    # Clean and tokenize query into keywords
    words = "".join(c if c.isalnum() or c.isspace() else " " for c in query.lower()).split()
    keywords = [w for w in words if w not in STOP_WORDS]
    if not keywords:
        return 0.0
        
    # Tokenize document words
    doc_words = set("".join(c if c.isalnum() or c.isspace() else " " for c in doc_text.lower()).split())
    
    # Calculate how many query keywords are in the document
    matches = sum(1 for kw in keywords if kw in doc_words)
    return matches / len(keywords)

def compute_phrase_boost(query: str, doc_text: str) -> float:
    """
    Looks for exact multi-word key phrases in the query and returns a boost if they are present in the chunk.
    """
    clean_query = query.lower()
    clean_doc = doc_text.lower()
    
    phrases = [
        "period of completion",
        "estimated cost",
        "earnest money",
        "earnest money deposit",
        "date of submission",
        "last date",
        "name of work",
        "technical capability",
        "financial requirement",
        "submission of bid"
    ]
    
    boost = 0.0
    for phrase in phrases:
        if phrase in clean_query and phrase in clean_doc:
            boost += 0.4  # Give a significant boost for matching the exact key phrase
            
    return boost

def run_db_experiment():
    # Force stdout to UTF-8 to handle Unicode characters (like ₹)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    # Path to our target tender PDF
    pdf_path = os.path.join(TENDERS_DIR, "tender.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find tender.pdf in Tenders folder: {pdf_path}")
        sys.exit(1)
        
    print("--- STEP 1: Extracting text from PDF ---")
    extracted_text = extract_text_from_pdf(pdf_path)
    print(f"Successfully extracted text from {len(extracted_text)} pages.")
    
    print("\n--- STEP 2: Creating text chunks ---")
    chunks = create_chunks(extracted_text)
    print(f"Generated {len(chunks)} chunks.")
    
    print("\n--- STEP 3: Initializing ChromaDB ---")
    client = get_chroma_client()
    collection = get_or_create_collection(client, "tender_requirements")
    print("Persistent Chroma client and collection initialized.")
    
    print("\n--- STEP 4: Storing chunks in ChromaDB (with automatic embeddings) ---")
    print("This will convert each chunk into a 384-dimensional vector and store it...")
    store_chunks_in_chroma(collection, chunks)
    print(f"Successfully stored all {len(chunks)} chunks in ChromaDB.")
    
    print("\n--- STEP 5: Interactive Semantic Search Chat ---")
    print("Type your questions below. Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            query = input("\nAsk a question about the tender: ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            break
            
        if not query:
            continue
            
        if query.lower() in ['exit', 'quit']:
            print("Exiting search session. Goodbye!")
            break
            
        # 1. Fetch a larger pool of candidate chunks (n_results=50) from Chroma
        results = query_chroma(collection, query, n_results=50)
        
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        # 2. Re-rank them using a Hybrid Score (Vector + Keyword + Phrase Boost)
        scored_candidates = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            # Clip distance between 0 and 1, then convert to similarity (1 = identical, 0 = opposite)
            sim = 1.0 - max(0.0, min(1.0, dist))
            
            # Compute exact keyword overlap score
            key_score = compute_keyword_score(query, doc)
            
            # Compute phrase matching boost
            phrase_boost = compute_phrase_boost(query, doc)
            
            # Calculate final combined score
            hybrid_score = 0.5 * sim + 0.5 * key_score + phrase_boost
            
            scored_candidates.append({
                "doc": doc,
                "meta": meta,
                "dist": dist,
                "sim": sim,
                "key_score": key_score,
                "phrase_boost": phrase_boost,
                "hybrid_score": hybrid_score
            })
            
        # Sort candidates by hybrid_score descending
        scored_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        # 3. Print the top 3 re-ranked results
        print(f"\nRetrieved and re-ranked top 3 matches (Hybrid Search):")
        for idx, cand in enumerate(scored_candidates[:3]):
            doc = cand["doc"]
            meta = cand["meta"]
            dist = cand["dist"]
            score = cand["hybrid_score"]
            
            # Determine match strength
            if score > 0.65:
                strength = "STRONG MATCH"
            elif score > 0.45:
                strength = "MODERATE MATCH"
            else:
                strength = "WEAK MATCH"
                
            print(f"\nResult #{idx + 1} | Page {meta['page_number']} | Hybrid Score: {score:.4f} (Dist: {dist:.4f}) | {strength}")
            print(f"--------------------------------------------------------------------------------")
            print(f"\"{doc}\"")
            print(f"--------------------------------------------------------------------------------")

if __name__ == "__main__":
    run_db_experiment()
