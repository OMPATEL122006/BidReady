import os
import re
from typing import List, Dict, Any
from app.models.document import DocumentType

class DocumentClassifier:
    """
    Automatic content-based document classifier that inspects text samples,
    structural headings, document extensions, and layout to identify the
    DocumentType (NIT, DETAILED_TENDER, BOQ, CORRIGENDUM, ADDENDUM, TECHNICAL_SPECIFICATION, OTHER)
    generically without hardcoded tender rules.
    """
    @classmethod
    def classify(cls, file_path: str, text_sample: str = "", tables: List[Dict[str, Any]] = None) -> DocumentType:
        fname = os.path.basename(file_path).lower()
        ext = os.path.splitext(file_path)[1].lower()
        sample_lower = (text_sample or "").lower()[:2000]

        # 1. Excel BOQ check
        if ext in [".xlsx", ".xls"]:
            return DocumentType.BOQ

        # 2. Filename keyword hints
        if any(w in fname for w in ["corrigendum", "amendment", "rectification"]):
            return DocumentType.CORRIGENDUM
        if "addendum" in fname:
            return DocumentType.ADDENDUM
        if "boq" in fname or "bill_of_quantities" in fname or "schedule_of_quantities" in fname:
            return DocumentType.BOQ
        if any(w in fname for w in ["tech_spec", "technical_spec", "specifications"]):
            return DocumentType.TECHNICAL_SPECIFICATION
        if any(w in fname for w in ["nit", "notice_inviting", "tendernotice"]):
            return DocumentType.NIT

        # 3. Content-based detection for Corrigendum / Addendum
        if any(re.search(r'\b' + re.escape(w) + r'\b', sample_lower) for w in ["corrigendum", "corrigenda", "amendment notice", "extension of date", "revised schedule"]):
            return DocumentType.CORRIGENDUM

        if "addendum" in sample_lower:
            return DocumentType.ADDENDUM

        # 4. Content-based detection for NIT (Notice Inviting Tender)
        nit_indicators = [
            "notice inviting tender", "press notice", "e-procurement notice",
            "invitation for bids", "ifb", "tender enquiry", "brief tender notice",
            "section i - notice inviting tender"
        ]
        if any(w in sample_lower for w in nit_indicators):
            return DocumentType.NIT

        # 5. Content-based detection for Technical Specification
        spec_indicators = [
            "technical specification", "scope of work", "technical clause",
            "particular specifications", "material specifications", "employer's requirements"
        ]
        if any(w in sample_lower for w in spec_indicators):
            return DocumentType.TECHNICAL_SPECIFICATION

        # 6. Detailed Tender Document check
        detailed_indicators = [
            "instructions to bidders", "conditions of contract", "general conditions",
            "special conditions", "qualification criteria", "form of bid",
            "contract data", "tender document"
        ]
        if any(w in sample_lower for w in detailed_indicators):
            return DocumentType.DETAILED_TENDER

        # Fallback based on text length: Short notices tend to be NIT, longer documents tend to be DETAILED_TENDER
        if len(text_sample or "") < 15000:
            return DocumentType.NIT
        else:
            return DocumentType.DETAILED_TENDER
