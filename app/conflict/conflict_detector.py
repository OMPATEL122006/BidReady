import re
from typing import List, Dict, Any, Optional
from app.models.retrieval import RetrievalResult, ConflictReport

class ConflictDetector:
    """
    Multi-document Conflict Detection & Authority Engine.
    Detects competing values for identical requested attributes across documents
    within a tender document set and evaluates document authority (Corrigendum vs NIT).
    """

    AMENDMENT_TERMS = [
        "corrigendum", "addendum", "revised", "extension", "extended",
        "amended", "supersedes", "replacement", "modification", "modified"
    ]

    @classmethod
    def detect_conflicts(cls, attribute: str, candidates: List[RetrievalResult]) -> ConflictReport:
        if not candidates or len(candidates) < 2 or attribute in ["general", "definition"]:
            return ConflictReport(has_conflict=False, requested_attribute=attribute)

        # Collect distinct evidence candidates by source document
        doc_sources: Dict[str, Dict[str, Any]] = {}
        for cand in candidates[:10]:
            doc_id = cand.chunk.document_id
            doc_name = cand.chunk.source_doc
            doc_type = str(cand.chunk.document_type).upper()
            doc_ver = cand.chunk.document_version
            text = cand.chunk.text

            if doc_id not in doc_sources:
                doc_sources[doc_id] = {
                    "document_id": doc_id,
                    "source_doc": doc_name,
                    "document_type": doc_type,
                    "document_version": doc_ver,
                    "page": cand.chunk.page_number,
                    "text": text,
                    "chunk": cand.chunk
                }

        if len(doc_sources) < 2:
            return ConflictReport(has_conflict=False, requested_attribute=attribute)

        # Extract values for attribute across document sources
        extracted_by_doc: List[Dict[str, Any]] = []
        for d_id, d_info in doc_sources.items():
            val = cls._extract_attribute_value(attribute, d_info["text"])
            if val:
                d_info["extracted_value"] = val
                extracted_by_doc.append(d_info)

        if len(extracted_by_doc) < 2:
            return ConflictReport(has_conflict=False, requested_attribute=attribute)

        # Compare values
        first_val = extracted_by_doc[0]["extracted_value"].lower().strip()
        has_divergence = any(d["extracted_value"].lower().strip() != first_val for d in extracted_by_doc[1:])

        if not has_divergence:
            return ConflictReport(has_conflict=False, requested_attribute=attribute)

        # Check if one document is a Corrigendum / Addendum that supersedes
        superseding_doc = None
        for d in extracted_by_doc:
            d_type = d["document_type"]
            text_lower = d["text"].lower()
            if d_type in ["CORRIGENDUM", "ADDENDUM"] or any(term in text_lower for term in cls.AMENDMENT_TERMS):
                superseding_doc = d
                break

        if superseding_doc:
            summary = (
                f"Conflicting values found across tender documents for '{attribute}'. "
                f"However, '{superseding_doc['source_doc']}' ({superseding_doc['document_type']}) "
                f"supersedes earlier documents with updated value: '{superseding_doc['extracted_value']}'."
            )
            return ConflictReport(
                has_conflict=True,
                requested_attribute=attribute,
                conflicting_sources=extracted_by_doc,
                superseded_by=superseding_doc,
                resolution_summary=summary
            )

        # Unresolved Conflict across documents!
        conflict_list_str = "; ".join([f"{d['source_doc']} (P{d['page']}): '{d['extracted_value']}'" for d in extracted_by_doc])
        summary = (
            f"⚠️ CONFLICT DETECTED across tender documents for '{attribute}'. "
            f"Competing values: {conflict_list_str}. "
            f"The available documents do not establish which value supersedes the other."
        )
        return ConflictReport(
            has_conflict=True,
            requested_attribute=attribute,
            conflicting_sources=extracted_by_doc,
            superseded_by=None,
            resolution_summary=summary
        )

    @classmethod
    def _extract_attribute_value(cls, attribute: str, text: str) -> Optional[str]:
        t_lower = text.lower()

        if attribute == "submission_deadline":
            m = re.search(r'(\d{1,2}[\./\-]\d{1,2}[\./\-]\d{2,4}(?:\s*at\s*\d{1,2}[:\.]\d{2}\s*(?:am|pm|hrs)?)?)', t_lower)
            if m:
                return m.group(1)

        elif attribute == "emd_amount":
            m = re.search(r'(?:rs\.?|₹)\s*\d+[\d,]*|\b\d+(?:\.\d+)?%\b|\b\d+[\d,]*\s*(?:lakh|crore|thousand|lac)\b', t_lower)
            if m:
                return m.group(0)

        elif attribute == "completion_period":
            m = re.search(r'\b\d+\s*(?:days|months|years|weeks)\b', t_lower)
            if m:
                return m.group(0)

        elif attribute == "bid_validity":
            m = re.search(r'\b\d+\s*\(?[a-z\s]*\)?\s*days\b', t_lower)
            if m:
                return m.group(0)

        elif attribute == "performance_security":
            m = re.search(r'\b\d+(?:\.\d+)?%\b', t_lower)
            if m:
                return m.group(0)

        return None
