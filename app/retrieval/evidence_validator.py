import re
from typing import Tuple


class EvidenceValidator:
    """
    Deterministic evidence gate.

    IMPORTANT:
    Retrieval relevance and answerability are separate concepts.

    This class NEVER says evidence is HIGH merely because:
    - the query words occur in the chunk
    - BM25 is high
    - vector similarity is high
    - the chunk is from the correct page
    """

    NEGATIVE = "🔴 LOW CONFIDENCE"
    PARTIAL = "🟡 MEDIUM CONFIDENCE"
    POSITIVE = "🟢 HIGH CONFIDENCE"

    def evaluate_answerability(
        self,
        query: str,
        requested_attribute: str,
        doc_text: str,
        score: float = 0.0,
    ) -> Tuple[bool, str]:

        q = self._clean(query)
        d = self._clean(doc_text)

        # Never allow a generic score to create HIGH confidence.
        # The evidence must contain a target-specific answer pattern.

        if self._is_emd_amount(q, requested_attribute):
            if self._has_emd_amount(d):
                return True, self.POSITIVE
            return False, self.NEGATIVE

        if self._is_emd_mode(q):
            if self._has_any(d, [
                "demand draft",
                "bankers cheque",
                "banker's cheque",
                "bank guarantee",
                "online payment",
                "rtgs",
                "neft",
                "pay order",
            ]):
                return True, self.POSITIVE
            return False, self.NEGATIVE

        if self._is_emd_beneficiary(q):
            if self._has_any(d, [
                "in favour of",
                "in favor of",
                "drawn in favour",
                "drawn in favor",
                "payable to",
            ]):
                return True, self.POSITIVE
            return False, self.NEGATIVE

        if self._is_emd_submission(q):
            if self._has_any(d, [
                "hard copy",
                "original emd",
                "physical copy",
                "submitted to",
                "deposit the emd",
                "emd should be submitted",
            ]) and self._has_date_or_deadline(d):
                return True, self.POSITIVE
            return False, self.NEGATIVE

        if self._is_completion(q, requested_attribute):
            if self._has_duration(d) and self._has_any(d, [
                "completion",
                "completed",
                "execution",
                "work shall be completed",
                "period of work",
            ]):
                return True, self.POSITIVE
            return False, self.NEGATIVE

        if self._is_bid_validity(q, requested_attribute):
            if self._has_duration(d) and "valid" in d:
                return True, self.POSITIVE
            return False, self.NEGATIVE

        if self._is_performance_security(q, requested_attribute):
            if self._has_percentage(d) and self._has_any(d, [
                "performance security",
                "performance guarantee",
            ]):
                return True, self.POSITIVE
            return False, self.NEGATIVE

        if self._is_security_deposit(q):
            if "security deposit" in d:
                return True, self.POSITIVE
            return False, self.NEGATIVE

        if self._is_financial_opening(q):
            if self._has_any(d, [
                "financial bid opening",
                "financial bid will be opened",
                "price bid opening",
                "price bid will be opened",
                "financial opening",
                "cover 2 opening",
            ]) and self._has_date_or_deadline(d):
                return True, self.POSITIVE
            return False, self.NEGATIVE

        if self._is_date_question(q, requested_attribute):
            # A date alone is NOT enough.
            # It must occur in a sentence containing the relevant concept.
            concept_groups = [
                ["submission", "bid"],
                ["opening", "bid"],
                ["publication", "tender"],
                ["issue", "tender"],
                ["clarification", "query"],
                ["pre-bid", "meeting"],
                ["start", "work"],
                ["letter of intent", "work order"],
                ["critical", "date"],
            ]

            for group in concept_groups:
                if all(term in d for term in group) and self._has_date_or_deadline(d):
                    return True, self.POSITIVE

            return False, self.NEGATIVE

        if self._is_yes_no_policy(q):
            policy_terms = [
                "allowed",
                "not allowed",
                "permitted",
                "not permitted",
                "prohibited",
                "mandatory",
                "required",
                "exempted",
                "exemption",
                "shall",
                "may",
            ]

            # Must have BOTH the subject and a policy statement.
            subject_terms = self._subject_terms(q)

            if any(term in d for term in subject_terms):
                if any(term in d for term in policy_terms):
                    return True, self.POSITIVE

            return False, self.NEGATIVE

        if self._is_contact_question(q):
            if ("@" in d or "email id" in d) and (
                "mobile" in d or "phone" in d
            ):
                return True, self.POSITIVE
            return False, self.NEGATIVE

        # Generic factual fallback.
        # Require multiple meaningful query concepts AND a concrete answer entity.
        q_terms = self._meaningful_terms(q)

        if len(q_terms) < 2:
            return False, self.NEGATIVE

        matched = sum(1 for term in q_terms if term in d)

        concrete = (
            self._has_percentage(d)
            or self._has_currency(d)
            or self._has_date_or_deadline(d)
            or self._has_duration(d)
            or self._has_any(d, [
                "required",
                "mandatory",
                "allowed",
                "not allowed",
                "permitted",
                "prohibited",
                "shall be",
                "is",
                "are",
            ])
        )

        if matched >= min(3, len(q_terms)) and concrete:
            return True, self.PARTIAL

        return False, self.NEGATIVE

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", str(text).lower()).strip()

    @staticmethod
    def _has_any(text: str, values) -> bool:
        return any(v in text for v in values)

    @staticmethod
    def _has_percentage(text: str) -> bool:
        return bool(
            re.search(r"\b\d+(?:\.\d+)?\s*%", text)
            or re.search(r"\b\d+(?:\.\d+)?\s*percent", text)
        )

    @staticmethod
    def _has_currency(text: str) -> bool:
        return bool(
            re.search(r"(₹|rs\.?|inr|rupees?)\s*[\d,]+(?:\.\d+)?", text)
            or re.search(
                r"\b[\d,]+(?:\.\d+)?\s*(?:lakh|lakhs|crore|crores|lac)\b",
                text,
            )
        )

    @staticmethod
    def _has_emd_amount(text: str) -> bool:
        return (
            ("emd" in text or "earnest money" in text)
            and (
                EvidenceValidator._has_currency(text)
                or EvidenceValidator._has_percentage(text)
            )
        )

    @staticmethod
    def _has_duration(text: str) -> bool:
        return bool(
            re.search(
                r"\b\d+(?:\.\d+)?\s*(?:days?|weeks?|months?|years?)\b",
                text,
            )
        )

    @staticmethod
    def _has_date_or_deadline(text: str) -> bool:
        return bool(
            re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", text)
            or re.search(
                r"\b(?:january|february|march|april|may|june|july|"
                r"august|september|october|november|december)\b",
                text,
            )
        )

    def _meaningful_terms(self, q: str):
        stop = {
            "what", "is", "the", "a", "an", "of", "for", "to",
            "in", "on", "at", "by", "and", "or", "are", "be",
            "this", "that", "which", "who", "when", "where",
            "how", "much", "does", "do", "can", "will", "should",
            "required", "applicable", "details", "period",
        }

        return [
            x for x in re.findall(r"[a-z0-9]+", q)
            if len(x) > 2 and x not in stop
        ]

    def _subject_terms(self, q: str):
        terms = self._meaningful_terms(q)

        aliases = {
            "jv": ["joint venture", "consortium"],
            "consortium": ["consortium", "joint venture"],
            "subcontracting": ["subcontract", "subcontracting"],
            "subcontract": ["subcontract", "subcontracting"],
            "msme": ["msme", "mse", "micro", "small enterprise"],
            "mse": ["mse", "msme", "micro", "small enterprise"],
            "blacklisting": ["blacklist", "blacklisted", "non-blacklisting"],
        }

        expanded = []
        for term in terms:
            expanded.extend(aliases.get(term, [term]))

        return list(set(expanded))

    def _is_emd_amount(self, q, attr):
        return (
            attr == "emd_amount"
            and "mode" not in q
            and "form" not in q
            and "favor" not in q
            and "favour" not in q
            and "drawn" not in q
        )

    def _is_emd_mode(self, q):
        return (
            ("emd" in q or "earnest money" in q)
            and any(x in q for x in ["mode", "form", "payment"])
        )

    def _is_emd_beneficiary(self, q):
        return (
            ("emd" in q or "earnest money" in q)
            and any(x in q for x in ["favor", "favour", "drawn"])
        )

    def _is_emd_submission(self, q):
        return (
            ("emd" in q or "earnest money" in q)
            and any(x in q for x in ["hard copy", "submit", "submission"])
        )

    def _is_completion(self, q, attr):
        return (
            attr == "completion_period"
            or "completion" in q
            or "execution time" in q
        )

    def _is_bid_validity(self, q, attr):
        return attr == "bid_validity" or "bid validity" in q

    def _is_performance_security(self, q, attr):
        return (
            attr == "performance_security"
            or "performance security" in q
            or "performance guarantee" in q
        )

    def _is_security_deposit(self, q):
        return "security deposit" in q

    def _is_financial_opening(self, q):
        return "financial" in q and "opening" in q

    def _is_date_question(self, q, attr):
        return (
            attr in {
                "submission_deadline",
                "opening_date",
                "critical_dates",
            }
            or any(x in q for x in [
                "deadline",
                "date",
                "when",
                "time",
                "scheduled",
            ])
        )

    def _is_yes_no_policy(self, q):
        return any(x in q for x in [
            "allowed",
            "permitted",
            "required",
            "mandatory",
            "applicable",
            "exemption",
            "exempt",
            "registration",
            "certificate",
            "subcontracting",
            "consortium",
            "joint venture",
            "blacklisting",
        ])

    def _is_contact_question(self, q):
        return any(x in q for x in [
            "contact person",
            "contact details",
            "phone",
            "mobile",
            "email",
        ])

    # Keep compatibility with existing HybridSearchEngine.
    def compute_domain_boost(self, query: str, doc_text: str) -> float:
        return 0.0

    def compute_boq_penalty(
        self,
        query: str,
        chunk_type: str,
        doc_text: str,
    ) -> float:
        return 0.0