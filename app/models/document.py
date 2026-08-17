from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional

class DocumentType(str, Enum):
    NIT = "NIT"
    DETAILED_TENDER = "DETAILED_TENDER"
    BOQ = "BOQ"
    TECHNICAL_SPECIFICATION = "TECHNICAL_SPECIFICATION"
    CORRIGENDUM = "CORRIGENDUM"
    ADDENDUM = "ADDENDUM"
    OTHER = "OTHER"

@dataclass
class DocumentMetadata:
    document_id: str
    file_name: str
    file_path: str
    file_type: str  # "pdf", "xlsx", "xls"
    tender_id: str = "default_tender"
    document_type: DocumentType = DocumentType.OTHER
    document_version: int = 1
    total_pages: int = 1
    is_scanned: bool = False
    ocr_applied_pages: List[int] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentPage:
    page_number: int
    text: str
    tables: List[Dict[str, Any]] = field(default_factory=list)
    boq_rows: List[Dict[str, Any]] = field(default_factory=list)
    is_ocr: bool = False

@dataclass
class TenderDocument:
    metadata: DocumentMetadata
    pages: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)

@dataclass
class TenderDocumentSet:
    tender_id: str
    documents: List[TenderDocument] = field(default_factory=list)

