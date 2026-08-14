import os
import sys

# Add current directory to path to resolve src imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.retriever import TenderRetriever
from src import config

# Define the comprehensive regression test suite of 22 key/problematic questions
REGRESSION_TESTS = [
    # Level 1 — Direct factual retrieval
    {
        "q_idx": 1,
        "question": "What is the name of the work?",
        "expected_pages": [1, 2, 3, 19, 26, 102]
    },
    {
        "q_idx": 2,
        "question": "Where is the work located?",
        "expected_pages": [15, 19, 26, 102]
    },
    {
        "q_idx": 3,
        "question": "What is the NIT number?",
        "expected_pages": [1, 3, 12]
    },
    {
        "q_idx": 4,
        "question": "What is the estimated cost of the tender?",
        "expected_pages": [1, 2, 12, 19, 28, 104]
    },
    {
        "q_idx": 8,
        "question": "What is the Earnest Money Deposit amount?",
        "expected_pages": [1, 3, 28]
    },
    {
        "q_idx": 9,
        "question": "What is the period of completion?",
        "expected_pages": [1, 3, 19]
    },
    {
        "q_idx": 10,
        "question": "What is the last date and time for submission of the bid?",
        "expected_pages": [1, 3, 13, 26]
    },
    {
        "q_idx": 11,
        "question": "When is the eligibility bid scheduled to be opened?",
        "expected_pages": [14, 26]
    },
    {
        "q_idx": 12,
        "question": "Who is inviting the tender?",
        "expected_pages": [12, 26]
    },
    {
        "q_idx": 13,
        "question": "Which government department does the tender belong to?",
        "expected_pages": [1, 9, 12, 26]
    },
    {
        "q_idx": 15,
        "question": "What type of tender is this?",
        "expected_pages": [12, 26]
    },
    {
        "q_idx": 16,
        "question": "What is the tender submission portal?",
        "expected_pages": [7]
    },
    {
        "q_idx": 17,
        "question": "What is the date of the NIT?",
        "expected_pages": [12, 13]
    },
    {
        "q_idx": 18,
        "question": "Who is the Assistant Engineer mentioned in the tender?",
        "expected_pages": [2, 3, 19, 26, 27]
    },
    {
        "q_idx": 19,
        "question": "Which Postal Civil Division is handling this tender?",
        "expected_pages": [1, 11, 12, 26]
    },
    {
        "q_idx": 20,
        "question": "What is the SH/head work?",
        "expected_pages": [1, 2, 19]
    },
    {
        "q_idx": 21,
        "question": "What types of contractors are allowed to submit bids?",
        "expected_pages": [12]
    },
    {
        "q_idx": 24,
        "question": "What is the required period within which similar works must have been completed?",
        "expected_pages": [12, 17]
    },
    {
        "q_idx": 27,
        "question": "What does the tender mean by 'similar work'?",
        "expected_pages": [12]
    },
    {
        "q_idx": 28,
        "question": "Which government organizations are mentioned as acceptable clients for similar works?",
        "expected_pages": [12, 13]
    },
    {
        "q_idx": 31,
        "question": "What happens if back-to-back execution is discovered?",
        "expected_pages": [12, 26]
    },
    {
        "q_idx": 34,
        "question": "What can a bidder submit if they do not have GST registration?",
        "expected_pages": [11]
    },
    {
        "q_idx": 37,
        "question": "Under what condition is the Integrity Pact required?",
        "expected_pages": [36]
    },
    {
        "q_idx": 38,
        "question": "What documents must be uploaded as part of the eligibility cover?",
        "expected_pages": [11, 17]
    },
    {
        "q_idx": 40,
        "question": "What is Form D used for?",
        "expected_pages": [17]
    },
    {
        "q_idx": 41,
        "question": "What are the acceptable forms of EMD?",
        "expected_pages": [10, 11, 13, 14]
    },
    {
        "q_idx": 47,
        "question": "What is the minimum portion of EMD that must be deposited in the prescribed form when using a bank guarantee?",
        "expected_pages": [14]
    },
    {
        "q_idx": 48,
        "question": "What is the required validity period of the bank guarantee portion of the EMD?",
        "expected_pages": [10, 14]
    },
    {
        "q_idx": 54,
        "question": "What percentage performance guarantee is mentioned in the tender?",
        "expected_pages": [9, 14, 28]
    },
    {
        "q_idx": 62,
        "question": "When is the financial cover opened?",
        "expected_pages": [11, 17, 21]
    },
    {
        "q_idx": 64,
        "question": "What happens if the uploaded documents do not match the physical documents submitted by the lowest bidder?",
        "expected_pages": [14, 15]
    },
    {
        "q_idx": 68,
        "question": "What happens if a bidder makes a post-tender modification outside the permitted process?",
        "expected_pages": [13]
    },
    {
        "q_idx": 69,
        "question": "How long must the tender remain open for acceptance?",
        "expected_pages": [9, 15, 21]
    }
]

def run_tests():
    # Force stdout to UTF-8 to handle Unicode characters
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("==================================================")
    print("      BIDREADY RETRIEVAL REGRESSION TESTS         ")
    print("==================================================")
    
    # 1. Initialize retriever
    retriever = TenderRetriever()
    
    # 2. Run Baseline (Reranker Disabled)
    print("\n--- Running Baseline (Reranker Disabled) ---")
    config.USE_RERANKER = False
    retriever.reranker.enabled = False
    
    baseline_top1_correct = 0
    baseline_top3_correct = 0
    baseline_results = []
    
    for t in REGRESSION_TESTS:
        results = retriever.retrieve(t["question"], n_results=3)
        top_pages = [r["metadata"]["page_number"] for r in results]
        
        is_top1_ok = len(top_pages) > 0 and top_pages[0] in t["expected_pages"]
        is_top3_ok = any(p in t["expected_pages"] for p in top_pages)
        
        if is_top1_ok:
            baseline_top1_correct += 1
        if is_top3_ok:
            baseline_top3_correct += 1
            
        baseline_results.append({
            "q_idx": t["q_idx"],
            "top_page": top_pages[0] if len(top_pages) > 0 else None,
            "top_pages": top_pages,
            "top1_ok": is_top1_ok,
            "top3_ok": is_top3_ok
        })
        
    # 3. Run Optimized (Reranker Enabled)
    print("\n--- Running Optimized (Reranker Enabled) ---")
    config.USE_RERANKER = True
    retriever.reranker.__init__() # Reload reranker
    
    opt_top1_correct = 0
    opt_top3_correct = 0
    opt_results = []
    
    for t in REGRESSION_TESTS:
        results = retriever.retrieve(t["question"], n_results=3)
        top_pages = [r["metadata"]["page_number"] for r in results]
        
        is_top1_ok = len(top_pages) > 0 and top_pages[0] in t["expected_pages"]
        is_top3_ok = any(p in t["expected_pages"] for p in top_pages)
        
        if is_top1_ok:
            opt_top1_correct += 1
        if is_top3_ok:
            opt_top3_correct += 1
            
        opt_results.append({
            "q_idx": t["q_idx"],
            "top_page": top_pages[0] if len(top_pages) > 0 else None,
            "top_pages": top_pages,
            "top1_ok": is_top1_ok,
            "top3_ok": is_top3_ok
        })

    # 4. Display Comparison Table
    print("\n==========================================================================================")
    print("                               REGRESSION TEST COMPARISON                                 ")
    print("==========================================================================================")
    print(f"{'Q#':<4} | {'Question':<55} | {'Expected':<10} | {'Base Top1':<9} | {'Opt Top1':<8} | {'Status':<7}")
    print("-" * 106)
    
    for idx, t in enumerate(REGRESSION_TESTS):
        base = baseline_results[idx]
        opt = opt_results[idx]
        
        status = "PASSED" if opt["top1_ok"] else "FAILED"
        if opt["top1_ok"] and not base["top1_ok"]:
            status = "FIXED 🟢"
        elif not opt["top1_ok"] and base["top1_ok"]:
            status = "REGRESSED 🔴"
            
        # Truncate question if too long
        q_text = t["question"]
        if len(q_text) > 52:
            q_text = q_text[:49] + "..."
            
        expected_str = ",".join(map(str, t["expected_pages"]))
        base_str = str(base["top_page"]) if base["top_page"] is not None else "-"
        opt_str = str(opt["top_page"]) if opt["top_page"] is not None else "-"
        
        print(f"Q{t['q_idx']:<2} | {q_text:<55} | {expected_str:<10} | {base_str:<9} | {opt_str:<8} | {status:<7}")
        
    print("==========================================================================================")
    
    # 5. Display Aggregated Statistics
    num_tests = len(REGRESSION_TESTS)
    print("\n--- METRIC SUMMARY ---")
    print(f"Total Tests: {num_tests}")
    print(f"Baseline Top-1 Accuracy: {baseline_top1_correct}/{num_tests} ({baseline_top1_correct/num_tests*100:.1f}%)")
    print(f"Baseline Top-3 Accuracy: {baseline_top3_correct}/{num_tests} ({baseline_top3_correct/num_tests*100:.1f}%)")
    print(f"Optimized Top-1 Accuracy: {opt_top1_correct}/{num_tests} ({opt_top1_correct/num_tests*100:.1f}%)")
    print(f"Optimized Top-3 Accuracy: {opt_top3_correct}/{num_tests} ({opt_top3_correct/num_tests*100:.1f}%)")
    
    fixed_count = sum(1 for b, o in zip(baseline_results, opt_results) if o["top1_ok"] and not b["top1_ok"])
    regressed_count = sum(1 for b, o in zip(baseline_results, opt_results) if not o["top1_ok"] and b["top1_ok"])
    print(f"False Positives Fixed: {fixed_count}")
    print(f"Regressions Introduced: {regressed_count}")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
