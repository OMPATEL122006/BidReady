from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional

class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    BOQ = "boq"

@dataclass
class Chunk:
    chunk_id: int
    text: str
    chunk_type: ChunkType
    page_number: int
    char_start: int
    char_end: int
    document_id: str
    source_doc: str
    tender_id: str = "default_tender"
    document_type: str = "OTHER"
    document_version: int = 1
    content_type: str = "text"
    sheet: Optional[str] = None
    row: Optional[int] = None
    section: Optional[str] = None
    clause: Optional[str] = None
    structured_json: Optional[str] = "{}"
    confidence: str = "🟡 MEDIUM CONFIDENCE"
    
    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "tender_id": str(self.tender_id),
            "document_id": str(self.document_id),
            "source_doc": str(self.source_doc),
            "document_type": str(self.document_type),
            "document_version": int(self.document_version),
            "content_type": str(self.content_type),
            "page_number": int(self.page_number),
            "char_start": int(self.char_start),
            "char_end": int(self.char_end),
            "chunk_type": str(self.chunk_type.value if isinstance(self.chunk_type, ChunkType) else self.chunk_type),
            "sheet": str(self.sheet or ""),
            "row": int(self.row or 0),
            "section": str(self.section or ""),
            "clause": str(self.clause or ""),
            "structured_json": str(self.structured_json or "{}")
        }

