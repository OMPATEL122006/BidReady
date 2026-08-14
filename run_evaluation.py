import os
import sys

# Add current directory to path to resolve src imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.retriever import TenderRetriever
from run_regression_tests import REGRESSION_TESTS

# List of 70 evaluation questions provided by the user
QUESTIONS = [
    # Level 1 — Direct factual retrieval
    "What is the name of the work?",
    "Where is the work located?",
    "What is the NIT number?",
    "What is the estimated cost of the tender?",
    "What is the civil component cost?",
    "What is the electrical component cost?",
    "What is the total estimated cost?",
    "What is the Earnest Money Deposit amount?",
    "What is the period of completion?",
    "What is the last date and time for submission of the bid?",
    "When is the eligibility bid scheduled to be opened?",
    "Who is inviting the tender?",
    "Which government department does the tender belong to?",
    "Which ministry does the Department of Posts belong to?",
    "What type of tender is this?",
    "What is the tender submission portal?",
    "What is the date of the NIT?",
    "Who is the Assistant Engineer mentioned in the tender?",
    "Which Postal Civil Division is handling this tender?",
    "What is the subject/head work mentioned after 'SH'?",
    
    # Level 2 — Specific requirement retrieval
    "What types of contractors are allowed to submit bids?",
    "Are Joint Ventures accepted in this tender?",
    "How many similar works are required for eligibility?",
    "What is the required period within which similar works must have been completed?",
    "How is the value of previously executed works adjusted to the current costing level?",
    "What annual enhancement rate is used for calculating the current value of completed works?",
    "What does the tender mean by 'similar work'?",
    "Which government organizations are mentioned as acceptable clients for similar works?",
    "Who must certify the bidder's performance for qualifying works?",
    "What affidavit must the bidder submit regarding back-to-back execution?",
    "What happens if back-to-back execution is discovered?",
    "Does the tender require an enlistment order?",
    "Is a GST registration certificate required?",
    "What can a bidder submit if they do not have GST registration?",
    "Is a PAN card required?",
    "Is an Integrity Pact required?",
    "Under what condition is the Integrity Pact required?",
    "What documents must be uploaded as part of the eligibility cover?",
    "What documents are included in Form C?",
    "What is Form D used for?",
    
    # Level 3 — EMD and financial conditions
    "What are the acceptable forms of EMD?",
    "Where must the original EMD be deposited?",
    "What receipt must the EMD receiving officer issue?",
    "What must be uploaded if the EMD is deposited through RTGS/NEFT?",
    "What happens if the same UTR is used for different tenders?",
    "Can part of the EMD be submitted through a bank guarantee?",
    "What is the minimum portion of EMD that must be deposited in the prescribed form when using a bank guarantee?",
    "What is the required validity period of the bank guarantee portion of the EMD?",
    "When is the EMD of unsuccessful bidders returned?",
    "What happens to the EMD of a bidder who withdraws the tender within seven days after the submission deadline?",
    "What happens if the bidder withdraws after those seven days?",
    "Through what method must a tender withdrawal be made?",
    "What happens to a bidder who withdraws and has their EMD forfeited?",
    "What percentage performance guarantee is mentioned in the tender?",
    "What determines whether the performance guarantee is based on the estimated cost or contract amount?",
    
    # Level 4 — Submission and procedural questions
    "What digital signature requirement must bidders satisfy?",
    "What file formats can contractors upload?",
    "How many electronic covers are used in the tender?",
    "What is the first electronic cover called?",
    "What is the second electronic cover called?",
    "Which cover is evaluated first?",
    "When is the financial cover opened?",
    "What happens if a bidder does not upload all required documents?",
    "What happens if the uploaded documents do not match the physical documents submitted by the lowest bidder?",
    "What happens if a bidder quotes NIL rates for every item?",
    "What happens if a bidder does not quote a percentage above or below the total amount in a percentage-rate tender?",
    "Can a bidder revise or withdraw the bid before the submission deadline?",
    "What happens if a bidder makes a post-tender modification outside the permitted process?",
    "How long must the tender remain open for acceptance?",
    "Within how many days must the successful bidder sign the contract from the stipulated start date"
]

def run_evaluation():
    # Force stdout to UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    print("--- Starting Optimized RAG Evaluation (70 Questions) ---")
    retriever = TenderRetriever()
    
    results_list = []
    
    for idx, query in enumerate(QUESTIONS):
        print(f"Evaluating Question {idx + 1}/{len(QUESTIONS)}: {query}")
        
        # Retrieve scored candidates in debug mode to extract component scores
        candidates = retriever.retrieve(query, debug=True)
        
        if candidates:
            results_list.append({
                "q_idx": idx + 1,
                "question": query,
                "top_result": candidates[0]
            })
        else:
            results_list.append({
                "q_idx": idx + 1,
                "question": query,
                "top_result": None
            })
            
    # Write report to markdown artifact file
    artifact_dir = "C:\\Users\\OMPATELL\\.gemini\\antigravity-ide\\brain\\b0bfca34-4791-48d5-b377-2fda0f334191"
    os.makedirs(artifact_dir, exist_ok=True)
    report_path = os.path.join(artifact_dir, "evaluation_results.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# BidReady Optimized RAG Evaluation Report\n\n")
        f.write("This report lists the retrieval results for the 70 standard evaluation questions run against the **Optimized Retrieval Engine** (Vector + BM25 + Phrase Boost + Domain Intent Boost + Cross-Encoder Reranker).\n\n")
        
        # Summary table
        f.write("## Retrieval Performance Summary\n\n")
        f.write("| Category | Score Range | Count |\n")
        f.write("| --- | --- | --- |\n")
        f.write("| **Total Questions** | N/A | 70 |\n")
        
        strong_matches = 0
        moderate_matches = 0
        weak_matches = 0
        
        for r in results_list:
            if r["top_result"]:
                conf = r["top_result"]["confidence"]
                if "🟢" in conf:
                    strong_matches += 1
                elif "🟡" in conf:
                    moderate_matches += 1
                else:
                    weak_matches += 1
                    
        f.write(f"| **High Confidence Matches (🟢 HIGH CONFIDENCE)** | Reranker > 0.1 & Passed Validation | {strong_matches} |\n")
        f.write(f"| **Medium Confidence Matches (🟡 MEDIUM CONFIDENCE)** | Reranker <= 0.1 & Passed Validation | {moderate_matches} |\n")
        f.write(f"| **Low Confidence Matches (🔴 LOW CONFIDENCE)** | Failed Validation (Fallback) | {weak_matches} |\n\n")
        
        f.write("## Detailed Results\n\n")
        
        for r in results_list:
            f.write(f"### Q{r['q_idx']}. {r['question']}\n\n")
            if r["top_result"]:
                res = r["top_result"]
                score = res["score"]
                dist = res["dist"]
                meta = res["metadata"]
                doc = res["text"]
                label = res["confidence"]
                    
                f.write(f"*   **Status:** {label}\n")
                f.write(f"*   **Final Combined Score:** `{score:.4f}`\n")
                f.write(f"    *   *Reranker Score:* `{res['rerank_score']:.4f}`\n")
                f.write(f"    *   *BM25 Score:* `{res['bm25_score']:.4f}` (Normalized: `{res['norm_bm25']:.4f}`)\n")
                f.write(f"    *   *Vector Sim (1-Dist):* `{res['sim']:.4f}` (Raw Dist: `{dist:.4f}`)\n")
                f.write(f"    *   *Phrase Boost:* `{res['phrase_boost']:.4f}`\n")
                f.write(f"    *   *Domain Intent Boost:* `{res['domain_boost']:.4f}`\n")
                f.write(f"    *   *Validation Penalty:* `{res.get('validation_penalty', 0.0):.4f}`\n")
                f.write(f"*   **Location:** Page `{meta['page_number']}`\n")
                f.write(f"*   **Retrieved Evidence Text:**\n")
                f.write(f"    > \"{doc}\"\n\n")
            else:
                f.write("*   **Status:** 🔴 NO RESULTS RETRIEVED\n\n")
            f.write("---\n\n")
            
    print(f"\nEvaluation completed! Report written to: {report_path}")

if __name__ == "__main__":
    run_evaluation()
