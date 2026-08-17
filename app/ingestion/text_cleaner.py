import re
from typing import Dict

class TextCleaner:
    """
    Post-OCR text sanitizer for cleaning common OCR noise and typos
    frequently found in government tender PDF scans.
    """
    # Mapping of common OCR misread patterns to correct terms
    TYPO_MAPPING: Dict[str, str] = {
        r'\broouns\b': 'rooms',
        r'\bTne\b': 'The',
        r'\btne\b': 'the',
        r'\benqulry\b': 'enquiry',
        r'\b11EST\b': 'IIEST',
        r'\b1IEST\b': 'IIEST',
        r'\b11est\b': 'iiest',
        r'\byaho0\b': 'yahoo',
        r'\banount\b': 'amount',
        r'\bnentionr\b': 'mention',
        r'\bfurnisned\b': 'furnished',
        r'\bShibpul\b': 'Shibpur',
        r'\bShibpui\b': 'Shibpur',
        r'\baliminium\b': 'aluminium',
        r'\bteuder\b': 'tender',
        r'\be-Tenders\b': 'e-Tenders',
        r'\be-Procurement\b': 'e-Procurement',
        r'\brequirement\b': 'requirement',
        r'\bconspicious\b': 'conspicuous',
        r'\bexemption\b': 'exemption',
        r'\bperformauce\b': 'performance',
        r'\bsercurity\b': 'security',
        r'\bvalldity\b': 'validity',
        r'\bsubnnission\b': 'submission',
        r'\bndate\b': 'date',
        r'\bnade\b': 'made',
        r'\benqlury\b': 'enquiry'
    }

    @classmethod
    def clean_text(cls, text: str) -> str:
        if not text:
            return ""

        cleaned = text
        for pattern, replacement in cls.TYPO_MAPPING.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        # Fix broken spaces before punctuation or digits
        cleaned = re.sub(r'\s+([.,;:!?])', r'\1', cleaned)
        cleaned = re.sub(r'(\d+)\s*%\s*', r'\1% ', cleaned)

        return cleaned
