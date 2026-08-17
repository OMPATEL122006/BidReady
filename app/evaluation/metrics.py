from dataclasses import dataclass
from typing import List, Optional
from app.models.retrieval import RetrievalResult
from app.evaluation.ground_truth import GroundTruthItem

@dataclass
class QuestionEvalResult:
    question_id: str
    question: str
    expected_pages: List[int]
    expected_answer: str
    rank_of_correct_chunk: int  # 1-indexed rank, or -1 if missed
    top_1_hit: bool
    top_3_hit: bool
    top_5_hit: bool
    mrr_score: float
    top_reranker_score: float
    retrieved_pages: List[int]
    status_label: str  # "✅ Rank #1", "⚠️ Rank #2", "⚠️ Rank #3", "❌ Missed"

class MetricsCalculator:
    """
    Evaluates retrieved candidate chunks against verified ground truth evidence.
    Calculates exact rank, Top-1, Top-3, Top-5 hit rates, and MRR.
    """
    @staticmethod
    def evaluate_question(gt: GroundTruthItem, candidates: List[RetrievalResult]) -> QuestionEvalResult:
        if not candidates:
            return QuestionEvalResult(
                question_id=gt.question_id,
                question=gt.question,
                expected_pages=gt.expected_pages,
                expected_answer=gt.expected_answer,
                rank_of_correct_chunk=-1,
                top_1_hit=False,
                top_3_hit=False,
                top_5_hit=False,
                mrr_score=0.0,
                top_reranker_score=0.0,
                retrieved_pages=[],
                status_label="❌ Missed"
            )

        top_reranker_score = candidates[0].rerank_score
        retrieved_pages = [c.chunk.page_number for c in candidates[:5]]

        rank_found = -1
        for idx, cand in enumerate(candidates, 1):
            text = cand.chunk.text
            page = cand.chunk.page_number
            
            # Check page match and keyphrase match
            page_match = (not gt.expected_pages) or (page in gt.expected_pages)
            keyphrase_match = any(kp.lower() in text.lower() for kp in gt.expected_keyphrases)

            if page_match and keyphrase_match:
                rank_found = idx
                break

        top_1 = (rank_found == 1)
        top_3 = (1 <= rank_found <= 3)
        top_5 = (1 <= rank_found <= 5)
        mrr = (1.0 / rank_found) if rank_found > 0 else 0.0

        if rank_found == 1:
            status = "✅ Rank #1"
        elif 1 < rank_found <= 3:
            status = f"⚠️ Rank #{rank_found}"
        elif 3 < rank_found <= 5:
            status = f"⚠️ Rank #{rank_found}"
        else:
            status = "❌ Missed"

        return QuestionEvalResult(
            question_id=gt.question_id,
            question=gt.question,
            expected_pages=gt.expected_pages,
            expected_answer=gt.expected_answer,
            rank_of_correct_chunk=rank_found,
            top_1_hit=top_1,
            top_3_hit=top_3,
            top_5_hit=top_5,
            mrr_score=mrr,
            top_reranker_score=top_reranker_score,
            retrieved_pages=retrieved_pages,
            status_label=status
        )
