import re
from typing import Dict, Any, List, Optional
from app.models.chunk import Chunk
from app.models.retrieval import RetrievalResult
from app.config.logging import logger

class AnswerabilityValidator:
    """
    Decoupled Evidence Answerability Validator.
    Evaluates strictly whether candidate evidence text contains the exact required
    answer information to answer the question, COMPLETELY INDEPENDENT of vector, BM25,
    or reranker relevance scores.
    """

    def validate_answerability(self, query: str, candidate_results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Validates Top candidate results and returns structured answerability report:
        {
            "answerable": "YES | PARTIAL | NO",
            "supporting_chunk": <chunk_id or null>,
            "reason": "...",
            "confidence": float (0.0 to 1.0)
        }
        """
        if not candidate_results:
            return {
                "answerable": "NO",
                "supporting_chunk": None,
                "reason": "No candidate evidence chunks retrieved.",
                "confidence": 0.0
            }

        q_clean = re.sub(r'[^\w\s]', '', query.lower())

        # Evaluate candidates in top rank order
        best_partial: Optional[Dict[str, Any]] = None

        for r in candidate_results:
            c = r.chunk
            d_text = c.text
            d_clean = d_text.lower()
            chunk_id = c.chunk_id

            eval_res = self._check_chunk_answerability(q_clean, d_clean, d_text)
            status = eval_res["status"]  # "YES", "PARTIAL", "NO"
            reason = eval_res["reason"]

            if status == "YES":
                return {
                    "answerable": "YES",
                    "supporting_chunk": chunk_id,
                    "reason": reason,
                    "confidence": 0.95
                }
            elif status == "PARTIAL" and best_partial is None:
                best_partial = {
                    "answerable": "PARTIAL",
                    "supporting_chunk": chunk_id,
                    "reason": reason,
                    "confidence": 0.60
                }

        if best_partial:
            return best_partial

        return {
            "answerable": "NO",
            "supporting_chunk": None,
            "reason": "Retrieved evidence does not contain the required factual answer for this question.",
            "confidence": 0.10
        }

    def evaluate_answerability(
        self,
        query: str,
        requested_attribute: str,
        doc_text: str,
        score: float
    ) -> Tuple[bool, str]:
        """
        Backward compatibility wrapper for single chunk answerability evaluation.
        """
        q_clean = re.sub(r'[^\w\s]', '', query.lower())
        d_clean = doc_text.lower()
        eval_res = self._check_chunk_answerability(q_clean, d_clean, doc_text)
        status = eval_res["status"]

        if status == "YES":
            return True, "🟢 HIGH CONFIDENCE"
        elif status == "PARTIAL":
            return True, "🟡 MEDIUM CONFIDENCE"
        else:
            return False, "🔴 LOW CONFIDENCE"

    def _check_chunk_answerability(self, q_clean: str, d_clean: str, d_raw: str) -> Dict[str, str]:
        """
        Generic, deterministic answerability verification per question intent category.
        Checks for presence of explicit requested entity, value, or statement.
        """
        # 1. Tender Type / Mode of Bidding (Q5)
        if any(k in q_clean for k in ["mode of bidding", "tender type", "type of tender", "bidding mode", "type of bid"]):
            has_bidding_mode = bool(re.search(
                r'e-tenders?|online|item\s*rate|percentage\s*rate|two\s*cover|single\s*cover|press\s*tender|open\s*tender|limited\s*tender|eproc|cppp',
                d_clean
            ))
            if has_bidding_mode:
                return {"status": "YES", "reason": "Evidence explicitly specifies the mode of bidding / tender submission type."}
            return {"status": "NO", "reason": "Evidence discusses tender requirements or turnover but lacks explicit mode of bidding / tender type."}

        # 2. Estimated Cost (Q6)
        if "estimated cost" in q_clean or "tender cost" in q_clean or "estimated value" in q_clean:
            has_cost_figure = bool(re.search(
                r'estimated\s*cost\s*:\s*(?:rs\.?|₹)?\s*\d+[\d,]*|estimated\s*value\s*:\s*(?:rs\.?|₹)?\s*\d+[\d,]*|cost\s*of\s*work\s*:\s*(?:rs\.?|₹)?\s*\d+[\d,]*',
                d_clean
            ))
            if has_cost_figure:
                return {"status": "YES", "reason": "Evidence explicitly states the numeric figure for estimated cost of work."}
            elif "estimated cost" in d_clean:
                return {"status": "PARTIAL", "reason": "Evidence mentions estimated cost clause but numeric estimated cost figure is absent or formula-only."}
            return {"status": "NO", "reason": "Evidence discusses turnover or general conditions but does not contain the estimated tender cost."}

        # 3. EMD Payment Modes (Q8)
        if ("emd" in q_clean or "earnest money" in q_clean) and ("mode" in q_clean or "form" in q_clean or "acceptable" in q_clean):
            has_emd_mode = bool(re.search(
                r'demand\s*draft|\bdd\b|bankers?\s*cheque|bank\s*guarantee|\bbg\b|online|rtgs|neft|pay\s*order|e-payment',
                d_clean
            ))
            if has_emd_mode:
                return {"status": "YES", "reason": "Evidence explicitly lists acceptable EMD payment modes (e.g. Demand Draft, Online, BG)."}
            elif "emd" in d_clean or "earnest money" in d_clean:
                return {"status": "PARTIAL", "reason": "Evidence mentions EMD rules or exemptions but does not state payment modes."}
            return {"status": "NO", "reason": "Evidence contains general contact info or terms without EMD payment mode details."}

        # 4. EMD Beneficiary / Demand Draft Favor (Q9)
        if ("emd" in q_clean or "earnest money" in q_clean or "demand draft" in q_clean) and ("favor" in q_clean or "favour" in q_clean or "drawn" in q_clean or "whom" in q_clean):
            has_favour_entity = bool(re.search(
                r'favour\s*of\s+[a-zA-Z0-9\s,\.]+|favor\s*of\s+[a-zA-Z0-9\s,\.]+|drawn\s*in\s*favour|payable\s*at\s+[a-zA-Z0-9\s]+',
                d_clean
            ))
            if has_favour_entity:
                return {"status": "YES", "reason": "Evidence explicitly identifies the entity/designation to whom the EMD demand draft must be drawn in favor of."}
            elif "emd" in d_clean or "demand draft" in d_clean:
                return {"status": "PARTIAL", "reason": "Evidence mentions EMD or Demand Draft but does not state the beneficiary entity."}
            return {"status": "NO", "reason": "Evidence contains contact details or terms without EMD beneficiary details."}

        # 5. EMD Submission Location / Deadline (Q10)
        if ("emd" in q_clean or "hard copy" in q_clean) and ("where" in q_clean or "when" in q_clean or "submission" in q_clean or "submitted" in q_clean or "reach" in q_clean):
            has_submission_details = bool(re.search(
                r'reach\s+the|submitted\s+to|submitted\s+in|reach\s+on\s+or\s+before|submitted\s+on\s+or\s+before|office\s+of',
                d_clean
            ))
            if has_submission_details:
                return {"status": "YES", "reason": "Evidence explicitly specifies where or by when the physical hard copy of EMD must be submitted."}
            elif "emd" in d_clean and ("submit" in d_clean or "deposit" in d_clean):
                return {"status": "PARTIAL", "reason": "Evidence states EMD deposit requirement but lacks explicit office location or physical submission deadline."}
            return {"status": "NO", "reason": "Evidence contains general contact info or terms without EMD submission location/deadline."}

        # 6. EMD Amount / Percentage (Q7)
        if ("emd" in q_clean or "earnest money" in q_clean) and ("amount" in q_clean or "percentage" in q_clean or "value" in q_clean):
            has_emd_val = bool(re.search(
                r'earnest\s*money\s*deposit\s*\(?emd\)?\s*:\s*(?:rs\.?|₹)?\s*\d+[\d,]*|emd\s*:\s*(?:rs\.?|₹)?\s*\d+[\d,]*|\b\d+(?:\.\d+)?\s*%\s*of\s*(?:the\s*)?estimated\s*cost',
                d_clean
            ))
            if has_emd_val:
                return {"status": "YES", "reason": "Evidence explicitly states the EMD monetary figure or percentage."}
            elif "emd" in d_clean or "earnest money" in d_clean:
                return {"status": "PARTIAL", "reason": "Evidence mentions EMD clause but exact amount/percentage is absent."}
            return {"status": "NO", "reason": "Evidence does not contain EMD amount or percentage."}

        # 7. Period of Completion (Q20)
        if "completion" in q_clean or "execution time" in q_clean or "period of completion" in q_clean:
            has_completion_period = bool(re.search(
                r'completed\s*within\s*\d+\s*(?:days?|months?|weeks?)|period\s*of\s*completion\s*:\s*\d+\s*(?:days?|months?|weeks?)|within\s*\d+\s*days?\s*of',
                d_clean
            ))
            if has_completion_period:
                return {"status": "YES", "reason": "Evidence explicitly states the time allowed for completion of work."}
            elif "completion" in d_clean or "liquidate damage" in d_clean:
                return {"status": "PARTIAL", "reason": "Evidence discusses completion or delay damages but does not state the explicit completion duration."}
            return {"status": "NO", "reason": "Evidence discusses turnover or general conditions but lacks completion period."}

        # 8. Contact Phone Number (Q67)
        if "phone" in q_clean or "mobile" in q_clean or "contact number" in q_clean:
            has_phone = bool(re.search(r'\b\d{8,12}\b|mobile\s*no|phone\s*no', d_clean))
            if has_phone:
                return {"status": "YES", "reason": "Evidence explicitly contains a phone or mobile contact number."}
            return {"status": "NO", "reason": "Evidence lacks a phone or mobile contact number."}

        # 9. Official Email Address (Q68)
        if "email" in q_clean:
            has_email = bool(re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|email\s*id', d_clean))
            if has_email:
                return {"status": "YES", "reason": "Evidence explicitly contains an official email address."}
            return {"status": "NO", "reason": "Evidence lacks an official email address."}

        # 10. General Question Fallback
        # Checks if key query nouns/verbs match AND evidence contains a factual figure or requirement statement
        stopwords = {
            "what", "is", "the", "of", "for", "this", "do", "we", "need", "in", "a", "an", "to",
            "how", "much", "are", "on", "at", "by", "or", "and", "be", "with", "from", "which",
            "required", "applicable", "allowed", "details", "submitted", "when", "where", "who", "any"
        }
        q_words = [w for w in q_clean.split() if w not in stopwords and len(w) > 2]
        matched_words = [w for w in q_words if w in d_clean]

        has_fact = bool(re.search(r'\b\d+(?:\.\d+)?%?\b|(?:rs\.?|₹)\s*\d+|\b(?:days|months|years|lakh|crore|nil|exempted|draft|cheque|submitted|mandatory|shall)\b', d_clean))

        if len(matched_words) >= 3 and has_fact:
            return {"status": "YES", "reason": "Evidence directly supports the question with matching keywords and factual figures/requirements."}
        elif len(matched_words) >= 2:
            return {"status": "PARTIAL", "reason": "Evidence discusses the query topic but lacks explicit factual answer details."}
        else:
            return {"status": "NO", "reason": "Evidence does not contain the required factual answer for this question."}
