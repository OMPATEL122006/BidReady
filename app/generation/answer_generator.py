import json
from typing import Dict, Any, List, Optional
from app.models.retrieval import RetrievalResult
from app.generation.groq_client import GroqClient
from app.retrieval.hybrid_search import HybridSearchEngine

class AnswerGenerator:
    """
    Combines hybrid retrieval with Groq LLM generation.
    Supports structured BOQ direct lookups and context truncation.
    """
    def __init__(self, search_engine: HybridSearchEngine = None, groq_client: GroqClient = None):
        self.search_engine = search_engine or HybridSearchEngine()
        self.groq_client = groq_client or GroqClient()

    def generate_answer(self, query: str, n_results: int = 3, document_id: Optional[str] = None, tender_id: Optional[str] = None) -> Dict[str, Any]:
        results = self.search_engine.search(query, n_results=n_results, document_id=document_id, tender_id=tender_id)
        if not results:
            return {
                "answer": "Not specified in the provided document.",
                "sources": [],
                "confidence": "🔴 LOW CONFIDENCE"
            }

        # Multi-Document Conflict Detection
        from app.conflict.conflict_detector import ConflictDetector
        query_model = self.search_engine.query_expander.expand_query(query)
        conflict_report = ConflictDetector.detect_conflicts(query_model.requested_attribute, results)

        # LLM Bypass for direct BOQ lookups
        top_res = results[0]
        top_chunk = top_res.chunk
        if top_chunk.chunk_type == "boq" and top_chunk.structured_json and not conflict_report.has_conflict:
            try:
                structured = json.loads(top_chunk.structured_json)
                item_no = structured.get("item_no", "")
                desc = structured.get("description", "")
                qty = structured.get("quantity", 0.0)
                unit = structured.get("unit", "")
                rate = structured.get("rate", 0.0)
                amt = structured.get("amount", 0.0)

                answer = (
                    f"According to the BOQ spreadsheet [{top_chunk.source_doc}], details for **Item {item_no}** are:\n"
                    f"- **Description:** {desc}\n"
                    f"- **Quantity:** {qty} {unit}\n"
                    f"- **Rate:** ₹ {rate:,.2f} per {unit if unit else 'unit'}\n"
                    f"- **Total Amount:** ₹ {amt:,.2f}"
                )
                if rate == 0.0 and amt == 0.0:
                    answer += "\n\n*(Note: The rate and amount are 0.0 in the template, indicating this is a blank item rate sheet to be quoted by the bidder).* [Page 1, BOQ]"
                else:
                    answer += f" [{top_chunk.source_doc}]"

                return {
                    "answer": answer,
                    "sources": [{"page": top_chunk.page_number, "text": top_chunk.text, "confidence": top_res.confidence}],
                    "confidence": "🟢 HIGH CONFIDENCE",
                    "conflict_report": conflict_report
                }
            except Exception:
                pass

        # Build formatted context
        context_parts = []
        sources = []
        max_confidence = "🔴 LOW CONFIDENCE"

        if conflict_report.has_conflict:
            context_parts.append(f"--- DOCUMENT CONFLICT NOTICE ---\n{conflict_report.resolution_summary}\n")

        for idx, res in enumerate(results):
            c = res.chunk
            page = c.page_number
            doc_info = f"{c.source_doc} ({c.document_type})"
            if "🟢" in res.confidence:
                max_confidence = "🟢 HIGH CONFIDENCE"
            elif "🟡" in res.confidence and "🟢" not in max_confidence:
                max_confidence = "🟡 MEDIUM CONFIDENCE"

            context_parts.append(f"--- Evidence Unit {idx + 1} ({doc_info}, Page {page}) ---\n{c.text}")
            sources.append({"doc": doc_info, "page": page, "text": c.text, "confidence": res.confidence})

        context_str = "\n\n".join(context_parts)
        if len(context_str) > 3500:
            context_str = context_str[:3500] + "\n[Context Truncated for token limit]"

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
        answer = self.groq_client.generate(user_prompt)

        return {
            "answer": answer,
            "sources": sources,
            "confidence": "⚠️ CONFLICT DETECTED" if conflict_report.has_conflict and not conflict_report.superseded_by else max_confidence,
            "conflict_report": conflict_report
        }
