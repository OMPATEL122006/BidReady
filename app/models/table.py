from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class TableCell:
    row_idx: int
    col_idx: int
    value: str
    is_header: bool = False

@dataclass
class TableData:
    headers: List[str]
    rows: List[List[str]]
    page_number: int
    sheet_name: str = ""
    markdown_representation: str = ""
