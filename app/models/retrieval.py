from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from app.models.chunk import Chunk

@dataclass
class QueryModel:
    original_query: str
    target_type: str = "GENERAL"  # DATE_TIME, CURRENCY_AMOUNT, URL, DEFINITION, CONSEQUENCE, BOQ
    requested_attribute: str = "general" # e.g. submission_deadline, emd_amount, completion_period, bid_validity, performance_security
    query_category: str = "FACTUAL" # FACTUAL vs CONCEPTUAL
    preferred_doc_types: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    tender_id: Optional[str] = None
    document_id: Optional[str] = None

@dataclass
class RetrievalResult:
    chunk: Chunk
    combined_score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    exact_match_score: float = 0.0
    rerank_score: float = 0.0
    confidence: str = "🟡 MEDIUM CONFIDENCE"

@dataclass
class ConflictReport:
    has_conflict: bool = False
    requested_attribute: str = "general"
    conflicting_sources: List[Dict[str, Any]] = field(default_factory=list)
    superseded_by: Optional[Dict[str, Any]] = None
    resolution_summary: str = "No conflict detected."

