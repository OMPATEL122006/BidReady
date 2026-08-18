import json
from typing import Dict, Any, List, Optional

from app.models.retrieval import RetrievalResult
from app.generation.groq_client import GroqClient
from app.retrieval.hybrid_search import HybridSearchEngine


class AnswerGenerator:
    """
    Combines hybrid retrieval with Groq LLM generation.

    Important:
    - Retrieval relevance is NOT the same as answerability.
    - LOW-confidence evidence is not used to generate the answer.
    - Final confidence comes only from evidence that passed validation.
    - Supports structured BOQ direct lookups.
    - Supports multi-document conflict detection.
    """

    def __init__(
        self,
        search_engine: HybridSearchEngine = None,
        groq_client: GroqClient = None
    ):
        self.search_engine = search_engine or HybridSearchEngine()
        self.groq_client = groq_client or GroqClient()

    def generate_answer(
        self,
        query: str,
        n_results: int = 5,
        document_id: Optional[str] = None,
        tender_id: Optional[str] = None
    ) -> Dict[str, Any]:

        # =========================================================
        # 1. RETRIEVE
        # =========================================================

        results = self.search_engine.search(
            query,
            n_results=n_results,
            document_id=document_id,
            tender_id=tender_id
        )

        if not results:
            return self._not_found_response([])

        # =========================================================
        # 2. QUERY MODEL / CONFLICT DETECTION
        # =========================================================

        from app.conflict.conflict_detector import ConflictDetector

        query_model = self.search_engine.query_expander.expand_query(
            query,
            document_id=document_id,
            tender_id=tender_id
        )

        conflict_report = ConflictDetector.detect_conflicts(
            query_model.requested_attribute,
            results
        )

        # =========================================================
        # 3. KEEP ONLY ANSWERABLE EVIDENCE
        # =========================================================

        answerable_results = [
            result
            for result in results
            if self._is_answerable(result)
        ]

        # Remove duplicate evidence
        answerable_results = self._deduplicate(
            answerable_results
        )

        # =========================================================
        # 4. NO SUPPORTING EVIDENCE
        # =========================================================

        if not answerable_results:

            return {
                "answer": (
                    "Not specified in the provided document."
                ),
                "sources": [
                    {
                        "doc": result.chunk.source_doc,
                        "page": result.chunk.page_number,
                        "text": result.chunk.text,
                        "confidence": result.confidence
                    }
                    for result in results[:3]
                ],
                "confidence": "🔴 LOW CONFIDENCE",
                "conflict_report": conflict_report
            }

        # =========================================================
        # 5. BOQ DIRECT LOOKUP
        # =========================================================

        top_res = answerable_results[0]
        top_chunk = top_res.chunk

        if (
            str(top_chunk.chunk_type).lower() == "boq"
            and top_chunk.structured_json
            and not conflict_report.has_conflict
        ):
            boq_answer = self._try_boq_answer(top_res)

            if boq_answer is not None:
                return {
                    "answer": boq_answer,
                    "sources": [
                        {
                            "doc": top_chunk.source_doc,
                            "page": top_chunk.page_number,
                            "text": top_chunk.text,
                            "confidence": top_res.confidence
                        }
                    ],
                    "confidence": "🟢 HIGH CONFIDENCE",
                    "conflict_report": conflict_report
                }

        # =========================================================
        # 6. BUILD EVIDENCE CONTEXT
        # =========================================================

        # Only send evidence that passed validation.
        # Maximum 5 chunks.
        answerable_results = answerable_results[:5]

        context_parts = []
        sources = []

        if conflict_report.has_conflict:
            context_parts.append(
                "--- DOCUMENT CONFLICT NOTICE ---\n"
                f"{conflict_report.resolution_summary}"
            )

        for idx, result in enumerate(
            answerable_results,
            start=1
        ):

            chunk = result.chunk

            doc_info = chunk.source_doc
            page = chunk.page_number

            context_parts.append(
                f"""
--- VERIFIED EVIDENCE {idx} ---
Document: {doc_info}
Page/Sheet: {page}
Document Type: {chunk.document_type}
Section: {chunk.section or "N/A"}
Clause: {chunk.clause or "N/A"}
Confidence: {result.confidence}

CONTENT:
{chunk.text}
--- END EVIDENCE {idx} ---
""".strip()
            )

            sources.append(
                {
                    "doc": doc_info,
                    "page": page,
                    "text": chunk.text,
                    "confidence": result.confidence
                }
            )

        context_str = "\n\n".join(context_parts)

        # Keep context controlled.
        if len(context_str) > 6000:
            context_str = (
                context_str[:6000]
                + "\n[Context truncated]"
            )

        # =========================================================
        # 7. STRICT GROQ PROMPT
        # =========================================================

        user_prompt = f"""
You are answering a government tender question.

QUESTION:
{query}

The following VERIFIED EVIDENCE was retrieved from the tender
documents.

IMPORTANT RULES:

1. Answer ONLY from the evidence below.
2. Do not use outside knowledge.
3. Do not invent missing values.
4. Do not infer an exact amount, percentage, date, time,
   name or requirement unless it is supported by the evidence.
5. A related sentence is NOT sufficient evidence.
6. If the evidence does not answer the question, say:
   "Not specified in the provided document."
7. Preserve exact numbers, dates, percentages and names.
8. Do not confuse:
   - EMD with Performance Security
   - Performance Security with Security Deposit
   - completion period with maintenance period
   - bid submission with bid opening
   - technical bid opening with financial bid opening
9. If multiple evidence units are required, combine them only
   when they clearly refer to the same tender.
10. Include the relevant page/source in the answer.

VERIFIED EVIDENCE:

{context_str}

Now answer the question concisely.
"""

        # =========================================================
        # 8. GROQ
        # =========================================================

        try:
            answer = self.groq_client.generate(
                user_prompt
            )

        except Exception as exc:

            return {
                "answer": (
                    "Unable to generate an answer "
                    "from the retrieved evidence."
                ),
                "sources": sources,
                "confidence": "🔴 LOW CONFIDENCE",
                "conflict_report": [
                    str(exc)
                ]
            }

        # =========================================================
        # 9. FINAL CONFIDENCE
        # =========================================================

        final_confidence = self._calculate_confidence(
            answerable_results
        )

        if (
            conflict_report.has_conflict
            and not conflict_report.superseded_by
        ):
            final_confidence = "⚠️ CONFLICT DETECTED"

        return {
            "answer": answer,
            "sources": sources,
            "confidence": final_confidence,
            "conflict_report": conflict_report
        }

    # =============================================================
    # HELPERS
    # =============================================================

    @staticmethod
    def _is_answerable(result: RetrievalResult) -> bool:
        """
        Evidence eligibility gate.

        HIGH/MEDIUM evidence can be passed to Groq.
        LOW evidence cannot.

        IMPORTANT:
        The retrieval score itself does NOT make evidence answerable.
        """

        confidence = str(
            getattr(result, "confidence", "")
        ).upper()

        return (
            "🟢 HIGH" in confidence
            or "🟡 MEDIUM" in confidence
            or "HIGH CONFIDENCE" in confidence
            or "MEDIUM CONFIDENCE" in confidence
        )

    @staticmethod
    def _calculate_confidence(results) -> str:

        if not results:
            return "🔴 LOW CONFIDENCE"

        confidences = [
            str(
                getattr(result, "confidence", "")
            ).upper()
            for result in results
        ]

        # HIGH only if supporting evidence itself is HIGH.
        if any(
            "🟢 HIGH" in confidence
            or "HIGH CONFIDENCE" in confidence
            for confidence in confidences
        ):
            return "🟢 HIGH CONFIDENCE"

        if any(
            "🟡 MEDIUM" in confidence
            or "MEDIUM CONFIDENCE" in confidence
            for confidence in confidences
        ):
            return "🟡 MEDIUM CONFIDENCE"

        return "🔴 LOW CONFIDENCE"

    @staticmethod
    def _deduplicate(results):

        seen = set()
        unique = []

        for result in results:

            chunk = result.chunk

            key = (
                str(chunk.source_doc),
                str(chunk.page_number),
                " ".join(
                    str(chunk.text)
                    .lower()
                    .split()
                )
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(result)

        return unique

    @staticmethod
    def _not_found_response(results):

        return {
            "answer": (
                "Not specified in the provided document."
            ),
            "sources": [
                {
                    "doc": result.chunk.source_doc,
                    "page": result.chunk.page_number,
                    "text": result.chunk.text,
                    "confidence": result.confidence
                }
                for result in results[:3]
            ],
            "confidence": "🔴 LOW CONFIDENCE"
        }

    @staticmethod
    def _try_boq_answer(result):

        chunk = result.chunk

        try:
            structured = json.loads(
                chunk.structured_json
            )

            item_no = structured.get(
                "item_no",
                ""
            )

            desc = structured.get(
                "description",
                ""
            )

            qty = structured.get(
                "quantity",
                0.0
            )

            unit = structured.get(
                "unit",
                ""
            )

            rate = structured.get(
                "rate",
                0.0
            )

            amount = structured.get(
                "amount",
                0.0
            )

            answer = (
                f"According to the BOQ spreadsheet "
                f"[{chunk.source_doc}], details for "
                f"**Item {item_no}** are:\n"
                f"- **Description:** {desc}\n"
                f"- **Quantity:** {qty} {unit}\n"
                f"- **Rate:** ₹ {rate:,.2f} per "
                f"{unit if unit else 'unit'}\n"
                f"- **Total Amount:** ₹ {amount:,.2f}"
            )

            if rate == 0.0 and amount == 0.0:
                answer += (
                    "\n\n*(Note: The rate and amount are "
                    "0.0 in the template, indicating this "
                    "is a blank item rate sheet to be quoted "
                    "by the bidder.)*"
                )

            return answer

        except Exception:
            return None