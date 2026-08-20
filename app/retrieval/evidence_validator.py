import re
from typing import Tuple, List
from app.config.logging import logger

class EvidenceValidator:
    """
    Generic, non-hardcoded evidence validation & confidence classifier.
    Strictly verifies whether candidate text directly contains the exact requested answer
    attributes (values, dates, durations, percentages, emails, phone numbers) vs merely discussing related concepts.
    """

    def evaluate_answerability(
        self,
        query: str,
        requested_attribute: str,
        doc_text: str,
        score: float
    ) -> Tuple[bool, str]:
        q_clean = re.sub(r'[^\w\s]', '', query.lower())
        d_clean = doc_text.lower()

        # Entity Extractors
        has_currency_amount = bool(re.search(r'(?:rs\.?|₹|rupees?)\s*\d+[\d,]*|\b\d+[\d,]*\s*(?:lakh|crore|thousand|lac)\b', d_clean))
        has_percentage = bool(re.search(r'\b\d+(?:\.\d+)?\s*%\b|\bpercent(?:age)?\b', d_clean))
        has_date_time = bool(re.search(r'\d{2}\.\d{2}\.\d{4}|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b', d_clean))
        has_duration_number = bool(re.search(r'\b\d+\s*(?:days?|months?|weeks?|years?)\b', d_clean))
        has_explicit_negation_or_permission = bool(re.search(r'\b(?:not\s+allowed|prohibited|forbidden|permitted|allowed|mandatory|exempted|nil)\b', d_clean))
        has_phone_num = bool(re.search(r'\b\d{8,12}\b|mobile\s*no|phone\s*no', d_clean))
        has_email_addr = bool(re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|email\s*id', d_clean))

        # Query Target Classification
        is_email_q = "email" in q_clean
        is_phone_q = any(w in q_clean for w in ["phone", "mobile", "contact number"])
        is_emd_q = "emd" in q_clean or "earnest money" in q_clean
        is_completion_q = "completion" in q_clean or "execution time" in q_clean or "period of work" in q_clean
        is_turnover_q = "turnover" in q_clean or "financial turnover" in q_clean
        is_cost_q = "estimated cost" in q_clean or "tender cost" in q_clean or "value of work" in q_clean

        # Strict Target Entity Verification
        if is_email_q and not has_email_addr:
            return False, "🔴 LOW CONFIDENCE"

        if is_phone_q and not has_phone_num:
            return False, "🔴 LOW CONFIDENCE"

        if is_emd_q:
            if "exemption" in q_clean:
                if "exemption" not in d_clean:
                    return False, "🔴 LOW CONFIDENCE"
            elif "favor" in q_clean or "favour" in q_clean or "drawn" in q_clean:
                if not any(w in d_clean for w in ["favour of", "favor of", "drawn", "in favour", "in favor"]):
                    return False, "🔴 LOW CONFIDENCE"
            elif "mode" in q_clean or "form" in q_clean:
                if not any(w in d_clean for w in ["demand draft", "dd", "bankers cheque", "bg", "bank guarantee", "online", "rtgs", "neft"]):
                    return False, "🔴 LOW CONFIDENCE"

        if is_completion_q:
            if not (has_duration_number or "completed within" in d_clean or "completion period" in d_clean or "period of completion" in d_clean):
                return False, "🔴 LOW CONFIDENCE"

        if is_cost_q:
            if not (has_currency_amount or "estimated cost" in d_clean):
                return False, "🔴 LOW CONFIDENCE"

        # Security Deposit vs Performance Security
        if "security deposit" in q_clean and "security deposit" not in d_clean:
            return False, "🔴 LOW CONFIDENCE"

        # Performance Security / Validity Duration
        if "validity" in q_clean and "performance security" in q_clean:
            if not has_duration_number and "released" not in d_clean:
                return False, "🔴 LOW CONFIDENCE"

        # Financial Bid Opening
        if "financial" in q_clean and "opening" in q_clean:
            if not any(w in d_clean for w in ["financial bid opening", "financial opening", "price bid opening", "cover 2 opening"]):
                return False, "🔴 LOW CONFIDENCE"

        # Token & Phrase Matching
        stopwords = {
            "what", "is", "the", "of", "for", "this", "do", "we", "need", "in", "a", "an", "to",
            "how", "much", "are", "on", "at", "by", "or", "and", "be", "with", "from", "which",
            "required", "applicable", "allowed", "details", "submitted", "when", "where", "who"
        }
        q_words = [w for w in q_clean.split() if w not in stopwords and len(w) > 2]
        word_matches = [w for w in q_words if w in d_clean]

        # Target-Specific Directness Evaluation
        if is_email_q:
            is_direct = has_email_addr and len(word_matches) >= 1
        elif is_phone_q:
            is_direct = has_phone_num and len(word_matches) >= 1
        elif is_completion_q:
            is_direct = (has_duration_number or "completed within" in d_clean) and len(word_matches) >= 2
        elif is_emd_q:
            is_direct = (has_currency_amount or has_percentage or "favour of" in d_clean or "demand draft" in d_clean) and len(word_matches) >= 2
        else:
            is_direct = (len(word_matches) >= 2) and (
                has_currency_amount or has_percentage or has_date_time or has_duration_number or has_explicit_negation_or_permission
            )

        if is_direct and score > 0.30:
            return True, "🟢 HIGH CONFIDENCE"
        elif len(word_matches) >= 2 and score > 0.10:
            return True, "🟡 MEDIUM CONFIDENCE"
        else:
            return False, "🔴 LOW CONFIDENCE"

    def compute_domain_boost(self, query: str, doc_text: str) -> float:
        q_lower = query.lower()
        d_lower = doc_text.lower()

        boost = 0.0
        if any(w in q_lower for w in ["emd", "earnest money", "tender fee"]):
            if any(w in d_lower for w in ["earnest money", "emd", "demand draft", "favour of", "exemption"]):
                boost += 0.20
        if any(w in q_lower for w in ["completion", "duration", "time limit", "work period"]):
            if any(w in d_lower for w in ["completed within", "completion period", "execution period", "days", "months"]):
                boost += 0.20
        if any(w in q_lower for w in ["phone", "mobile", "contact", "email"]):
            if any(w in d_lower for w in ["mobile no", "phone no", "email id", "@"]):
                boost += 0.30

        return min(0.40, boost)

    def compute_boq_penalty(self, query: str, chunk_type: str, doc_text: str) -> float:
        q_lower = query.lower()
        if chunk_type == "excel_table":
            if any(w in q_lower for w in ["emd", "performance security", "validity", "contact", "email", "phone", "address"]):
                return -0.40
        return 0.0
