import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.pipeline import TenderRAGPipeline
from scripts.validate_retrieval import QUESTION_SUITE, scan_available_files

# Ground-truth Category A (42 answerable questions with evidence chunk keyphrases)
# and Category B (28 absent-clause questions)
CATEGORY_A_MAPPING = {
    1: ["repair and renovation of rooms at soms", "repair and renovation"],
    2: ["cwsoms_27072026", "668r"],
    3: ["iiest", "shibpur", "indian institute of engineering science"],
    4: ["soms", "shibpur"],
    5: ["e-tenders", "online", "central public procurement portal"],
    6: ["30%", "estimated cost"],
    7: ["1%", "estimated cost", "emd"],
    8: ["demand draft", "the registrar, iiest, shibpur"],
    9: ["the registrar, iiest, shibpur"],
    10: ["e-procurement cell", "16.08.2026"],
    11: ["exemption will be provided", "valid documents"],
    12: ["5%", "performance security"],
    14: ["period of maintenance has passed"],
    16: ["16.08.2026", "12:00 pm"],
    17: ["17.08.2026", "12:30 pm"],
    19: ["90", "ninety", "days"],
    20: ["30 days"],
    22: ["27.07.2026"],
    25: ["16.08.2026", "17.08.2026"],
    26: ["last seven (07) years", "40%"],
    27: ["40%", "estimated cost"],
    28: ["50%", "estimated cost"],
    29: ["80%", "estimated cost"],
    30: ["30%", "last 3 years"],
    32: ["joint ventures", "not accepted"],
    34: ["valid trade license"],
    36: ["registered under gst"],
    37: ["permanent account number", "pan"],
    40: ["financial turnover", "last 3 years"],
    41: ["undertaking of non-blacklisted"],
    42: ["inspect the site"],
    43: ["mandatory information to be furnished"],
    46: ["period of maintenance", "twelve months"],
    47: ["responsible for rectifying any defects"],
    56: ["no advance payment"],
    59: ["all amounts quoted should be inclusive"],
    65: ["liquidate damage", "0.5 percent"],
    66: ["mr. dibyendu banerjee"],
    67: ["9434114888"],
    68: ["dibban2003@yahoo.co.in"],
    69: ["soms", "iiest"],
    70: ["central public procurement portal"]
}

CATEGORY_B_INDICES = [13, 15, 18, 21, 23, 24, 31, 33, 35, 38, 39, 44, 45, 48, 49, 50, 51, 52, 53, 54, 55, 57, 58, 60, 61, 62, 63, 64]

def main():
    print("=================================================================")
    print("  BIDREADY DIAGNOSTIC EVALUATION: CATEGORY A & B BENCHMARK       ")
    print("=================================================================\n")

    available_files = scan_available_files()
    shibpur_files = [f for f in available_files if "shibpur" in f.lower()]

    if not shibpur_files:
        print("Error: Shibpur tender files not found in workspace.")
        return

    pipeline = TenderRAGPipeline()
    pipeline.clear()

    tender_id = pipeline.ingest_tender_set(shibpur_files)

    output_lines = []
    output_lines.append("=================================================================")
    output_lines.append("   CATEGORY A & CATEGORY B DIAGNOSTIC EVALUATION REPORT          ")
    output_lines.append("=================================================================\n")

    # Metrics trackers for Category A
    cat_a_total = len(CATEGORY_A_MAPPING)
    r1_hits = 0
    r3_hits = 0
    r5_hits = 0
    r10_hits = 0
    mrr_sum = 0.0

    # Metrics trackers for Category B
    cat_b_total = len(CATEGORY_B_INDICES)
    cat_b_false_pass = 0

    print("--- CATEGORY A (42 ANSWERABLE QUESTIONS) EVALUATION ---\n")
    output_lines.append("--- CATEGORY A (42 ANSWERABLE QUESTIONS) EVALUATION ---\n")

    for q_idx in sorted(CATEGORY_A_MAPPING.keys()):
        question = QUESTION_SUITE[q_idx - 1]
        keyphrases = CATEGORY_A_MAPPING[q_idx]

        results = pipeline.search_engine.search(question, n_results=10, tender_id=tender_id)

        hit_rank = 0
        for rank, r in enumerate(results, 1):
            text_lower = r.chunk.text.lower()
            if any(kp in text_lower for kp in keyphrases):
                hit_rank = rank
                break

        if hit_rank == 1:
            r1_hits += 1
        if hit_rank > 0 and hit_rank <= 3:
            r3_hits += 1
        if hit_rank > 0 and hit_rank <= 5:
            r5_hits += 1
        if hit_rank > 0 and hit_rank <= 10:
            r10_hits += 1

        if hit_rank > 0:
            mrr_sum += (1.0 / hit_rank)

        rank_str = f"#{hit_rank}" if hit_rank > 0 else "NOT IN TOP 10"
        line = f"Q{q_idx:02d} [Cat A] | Question: {question[:40]}... | Evidence Rank: {rank_str}"
        print(line)
        output_lines.append(line)

    print("\n--- CATEGORY B (28 ABSENT-CLAUSE QUESTIONS) EVALUATION ---\n")
    output_lines.append("\n--- CATEGORY B (28 ABSENT-CLAUSE QUESTIONS) EVALUATION ---\n")

    for q_idx in CATEGORY_B_INDICES:
        question = QUESTION_SUITE[q_idx - 1]
        results = pipeline.search_engine.search(question, n_results=5, tender_id=tender_id)

        # Check how many candidates pass answerability gate (HIGH or MEDIUM confidence)
        passed_candidates = [r for r in results if "🟢" in r.confidence or "🟡" in r.confidence]

        if passed_candidates:
            cat_b_false_pass += 1
            status_str = f"REACHES LLM (FALSE POSITIVE ⚠️) - {len(passed_candidates)} candidate(s) passed"
        else:
            status_str = "REJECTED (SAFE 🟢) - Filtered to 'Not specified'"

        line = f"Q{q_idx:02d} [Cat B] | Question: {question[:40]}... | Status: {status_str}"
        print(line)
        output_lines.append(line)

    # Calculate overall summary metrics
    rec1 = (r1_hits / cat_a_total) * 100.0
    rec3 = (r3_hits / cat_a_total) * 100.0
    rec5 = (r5_hits / cat_a_total) * 100.0
    rec10 = (r10_hits / cat_a_total) * 100.0
    mrr = mrr_sum / cat_a_total

    false_pass_pct = (cat_b_false_pass / cat_b_total) * 100.0
    safe_reject_pct = 100.0 - false_pass_pct

    summary_str = (
        f"\n=================================================================\n"
        f"                  OVERALL EVALUATION METRICS SUMMARY             \n"
        f"=================================================================\n"
        f"  CATEGORY A METRICS (42 Answerable Questions):\n"
        f"    - Recall@1  : {rec1:.1f}%  ({r1_hits}/{cat_a_total})\n"
        f"    - Recall@3  : {rec3:.1f}%  ({r3_hits}/{cat_a_total})\n"
        f"    - Recall@5  : {rec5:.1f}%  ({r5_hits}/{cat_a_total})\n"
        f"    - Recall@10 : {rec10:.1f}%  ({r10_hits}/{cat_a_total})\n"
        f"    - MRR       : {mrr:.4f}\n\n"
        f"  CATEGORY B METRICS (28 Absent-Clause Questions):\n"
        f"    - Correctly Filtered (Safe 'Not specified'): {cat_b_total - cat_b_false_pass}/{cat_b_total} ({safe_reject_pct:.1f}%)\n"
        f"    - Incorrectly Reaches LLM (False Positives)  : {cat_b_false_pass}/{cat_b_total} ({false_pass_pct:.1f}%)\n"
        f"================================================================="
    )

    print(summary_str)
    output_lines.append(summary_str)

    out_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "category_ab_evaluation_report.txt")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"\n[DONE] Category A & B evaluation complete! Saved to: {out_file}")

if __name__ == "__main__":
    main()
