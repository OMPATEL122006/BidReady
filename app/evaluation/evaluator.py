import os
import json
import time
from typing import List, Dict, Any
from app.pipeline import TenderRAGPipeline
from app.evaluation.ground_truth import SHIBPUR_GROUND_TRUTH, GroundTruthItem
from app.evaluation.metrics import MetricsCalculator, QuestionEvalResult
from app.config.logging import logger

class RetrievalEvaluator:
    """
    Evaluates retrieval & reranking accuracy against ground truth evidence,
    generating an objective Evaluation Dashboard and JSON benchmark matrix.
    """
    def __init__(self, pipeline: TenderRAGPipeline = None):
        self.pipeline = pipeline or TenderRAGPipeline()

    def run_evaluation(self, output_dir: str = None) -> Dict[str, Any]:
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Running Ground Truth Evaluation across {len(SHIBPUR_GROUND_TRUTH)} benchmark questions...")
        start_time = time.time()

        eval_results: List[QuestionEvalResult] = []
        for idx, gt in enumerate(SHIBPUR_GROUND_TRUTH, 1):
            print(f"[{idx}/{len(SHIBPUR_GROUND_TRUTH)}] Evaluating {gt.question_id}: {gt.question[:45]}...", flush=True)
            candidates = self.pipeline.search_engine.search(gt.question, n_results=5)
            res = MetricsCalculator.evaluate_question(gt, candidates)
            eval_results.append(res)

        elapsed_sec = time.time() - start_time
        total_q = len(eval_results)
        top1_hits = sum(1 for r in eval_results if r.top_1_hit)
        top3_hits = sum(1 for r in eval_results if r.top_3_hit)
        top5_hits = sum(1 for r in eval_results if r.top_5_hit)
        avg_mrr = sum(r.mrr_score for r in eval_results) / total_q if total_q > 0 else 0.0

        top1_acc = (top1_hits / total_q * 100.0) if total_q > 0 else 0.0
        top3_acc = (top3_hits / total_q * 100.0) if total_q > 0 else 0.0
        top5_acc = (top5_hits / total_q * 100.0) if total_q > 0 else 0.0

        dashboard_md = self._build_markdown_dashboard(
            eval_results, total_q, top1_hits, top3_hits, top5_hits, top1_acc, top3_acc, top5_acc, avg_mrr, elapsed_sec
        )

        md_path = os.path.join(output_dir, "retrieval_eval_dashboard.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(dashboard_md)

        json_path = os.path.join(output_dir, "retrieval_eval_matrix.json")
        json_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": total_q,
            "top_1_accuracy_pct": round(top1_acc, 2),
            "top_3_accuracy_pct": round(top3_acc, 2),
            "top_5_accuracy_pct": round(top5_acc, 2),
            "mrr_score": round(avg_mrr, 4),
            "evaluation_time_sec": round(elapsed_sec, 2),
            "itemized_results": [
                {
                    "question_id": r.question_id,
                    "question": r.question,
                    "expected_pages": r.expected_pages,
                    "expected_answer": r.expected_answer,
                    "rank_of_correct_chunk": r.rank_of_correct_chunk,
                    "top_1_hit": r.top_1_hit,
                    "top_3_hit": r.top_3_hit,
                    "top_5_hit": r.top_5_hit,
                    "top_reranker_score": round(r.top_reranker_score, 4),
                    "retrieved_pages": r.retrieved_pages,
                    "status": r.status_label
                }
                for r in eval_results
            ]
        }
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(json_data, indent=2))

        logger.info(f"Evaluation complete! Top-1 Acc: {top1_acc:.1f}% | Top-3 Acc: {top3_acc:.1f}% | MRR: {avg_mrr:.4f}")
        logger.info(f"Dashboard saved to: {md_path}")

        return json_data

    def _build_markdown_dashboard(
        self,
        results: List[QuestionEvalResult],
        total_q: int,
        top1_hits: int,
        top3_hits: int,
        top5_hits: int,
        top1_acc: float,
        top3_acc: float,
        top5_acc: float,
        avg_mrr: float,
        elapsed_sec: float
    ) -> str:
        lines = []
        lines.append("# 📊 Retrieval Evaluation Dashboard\n")
        lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
        lines.append(f"**Evaluation Time:** {elapsed_sec:.2f} seconds\n")
        lines.append("## 🏆 Summary Benchmark Metrics\n")
        lines.append(f"- **Top-1 Retrieval Accuracy:** `{top1_acc:.1f}%` ({top1_hits}/{total_q})")
        lines.append(f"- **Top-3 Retrieval Accuracy:** `{top3_acc:.1f}%` ({top3_hits}/{total_q})")
        lines.append(f"- **Top-5 Retrieval Accuracy:** `{top5_acc:.1f}%` ({top5_hits}/{total_q})")
        lines.append(f"- **Mean Reciprocal Rank (MRR):** `{avg_mrr:.4f}`\n")
        lines.append("## 📋 Itemized Retrieval & Ranking Performance Matrix\n")
        lines.append("| Q# | Audit Question | Target Page(s) | Correct Chunk Rank | Top-1 | Top-3 | Top-5 | Top Reranker Score | Match Status | Expected Ground Truth Answer |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")

        for r in results:
            t1 = "✅" if r.top_1_hit else "❌"
            t3 = "✅" if r.top_3_hit else "❌"
            t5 = "✅" if r.top_5_hit else "❌"
            pages_str = ", ".join([f"P{p}" for p in r.expected_pages]) if r.expected_pages else "Any"
            rank_str = f"**Rank #{r.rank_of_correct_chunk}**" if r.rank_of_correct_chunk > 0 else "**Missed**"
            safe_ans = r.expected_answer.replace("|", "/")

            lines.append(f"| {r.question_id} | {r.question} | {pages_str} | {rank_str} | {t1} | {t3} | {t5} | `{r.top_reranker_score:.4f}` | {r.status_label} | {safe_ans} |")

        return "\n".join(lines)
