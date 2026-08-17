import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.pipeline import TenderRAGPipeline
from app.evaluation.evaluator import RetrievalEvaluator
from app.config.settings import TENDERS_DIR

def main():
    print("=================================================================")
    print("        BIDREADY RETRIEVAL EVALUATION DASHBOARD GENERATOR        ")
    print("=================================================================")

    pipeline = TenderRAGPipeline()
    pipeline.clear()

    # Find Shibpur tender files by default
    shibpur_dir = os.path.join(TENDERS_DIR, "Shibpur")
    files_to_ingest = []

    if os.path.exists(shibpur_dir):
        for root, _, files in os.walk(shibpur_dir):
            for file in files:
                if file.lower().endswith((".pdf", ".xlsx", ".xls")):
                    files_to_ingest.append(os.path.join(root, file))

    if not files_to_ingest:
        for root, _, files in os.walk(TENDERS_DIR):
            for file in files:
                if file.lower().endswith((".pdf", ".xlsx", ".xls")):
                    files_to_ingest.append(os.path.join(root, file))

    print(f"Indexing {len(files_to_ingest)} benchmark document(s)...")
    for fp in files_to_ingest:
        pipeline.ingest_file(fp)

    evaluator = RetrievalEvaluator(pipeline=pipeline)
    summary_data = evaluator.run_evaluation()

    print("\n=================================================================")
    print("📊 BENCHMARK RETRIEVAL EVALUATION RESULTS")
    print("=================================================================")
    print(f"Total Questions Evaluated: {summary_data['total_questions']}")
    print(f"Top-1 Accuracy:            {summary_data['top_1_accuracy_pct']}%")
    print(f"Top-3 Accuracy:            {summary_data['top_3_accuracy_pct']}%")
    print(f"Top-5 Accuracy:            {summary_data['top_5_accuracy_pct']}%")
    print(f"Mean Reciprocal Rank (MRR):{summary_data['mrr_score']}")
    print("=================================================================")
    print(f"Dashboard saved to: outputs/retrieval_eval_dashboard.md")
    print(f"JSON matrix saved to: outputs/retrieval_eval_matrix.json")
    print("=================================================================\n")

if __name__ == "__main__":
    main()
