from dataclasses import dataclass
from typing import Optional

@dataclass
class BOQItem:
    item_no: str
    description: str
    quantity: float
    unit: str
    rate: float
    amount: float
    code: Optional[str] = ""
    page_number: int = 1
    sheet_name: str = ""
