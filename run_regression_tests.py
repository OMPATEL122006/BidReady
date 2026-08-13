import os
import sys

# Add current directory to path to resolve src imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.retriever import TenderRetriever
from src import config

# Define the regression test suite of 14 problematic/verification questions
# Maps Question -> List of acceptable correct page numbers containing the actual answer
REGRESSION_TESTS = [
    {
        "q_idx": 3,
        "question": "What is the NIT number?",
        "expected_pages": [1, 3, 12] # NIT No. : 17/AE/PCSD/LKO/26-27 is on pages 1, 3, and 12
    },
    {
        "q_idx": 8,
        "question": "What is the EMD amount?",
        "expected_pages": [1, 3, 28] # Rs. 4,163/- is on pages 1, 3, and 28
    },
    {
        "q_idx": 9,
        "question": "What is the period of completion?",
        "expected_pages": [1, 3, 19] # 01 (One) Month is on pages 1, 3, and 19
    },
    {
        "q_idx": 10,
        "question": "What is the last date and time for submission?",
        "expected_pages": [1, 3, 13] # 20.08.2026 upto 11.00 hrs is on pages 1, 3, and 13
    },
    {
        "q_idx": 11,
        "question": "When is the eligibility bid scheduled to be opened?",
        "expected_pages": [14, 26] # Opened on 21.08.2026 at 11.00 hrs is on pages 14 and 26
    },
    {
        "q_idx": 17,
        "question": "What is the date of the NIT?",
        # The NIT does not have a formal publication date printed, but page 13 mentions 14.08.2026 and page 12 mentions original date
        "expected_pages": [12, 13]
    },
    {
        "q_idx": 24,
        "question": "What is the required period within which similar works must have been completed?",
        "expected_pages": [12, 17] # Last 7 years is mentioned on pages 12 and 17
    },
    {
        "q_idx": 27,
        "question": "What does 'similar work' mean?",
        "expected_pages": [12] # Definition is explicitly on page 12
    },
    {
        "q_idx": 34,
        "question": "What can a bidder submit if they do not have GST registration?",
        "expected_pages": [11] # GST undertaking text is on page 11
    },
    {
        "q_idx": 41,
        "question": "What are the acceptable forms of EMD?",
        "expected_pages": [10, 13, 14] # Forms (DD, FDR, BG, etc.) are detailed on pages 10, 13, and 14
    },
    {
        "q_idx": 48,
        "question": "What is the validity period of the bank guarantee portion of the EMD?",
        "expected_pages": [10, 14] # 180 days or more is on pages 10 and 14
    },
    {
        "q_idx": 62,
        "question": "When is the financial cover opened?",
        "expected_pages": [11, 17, 21] # "at a later date" or "at notified time" on pages 11, 17, and 21
    },
    # Verification questions that were correct but had low scores
    {
        "q_idx": 20,
        "question": "What is the SH/head work?",
        "expected_pages": [1, 2, 19] # Dismantaling of cycle stand and store is on pages 1, 2, and 19
    },
    {
        "q_idx": 31,
        "question": "What happens if back-to-back execution is discovered?",
        "expected_pages": [12, 26] # Debarred forever or forfeiture of EMD/PG on pages 12 and 26
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
    
    # 2. Run Baseline (Without Reranker)
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
        
    # 3. Run Optimized (With Reranker if available)
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
