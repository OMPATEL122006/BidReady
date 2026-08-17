import os
from typing import Dict, List, Any
import pdfplumber
from app.ingestion.ocr import OCREngine
from app.config.logging import logger

class PDFParser:
    """
    Parser for native text PDFs and scanned document PDFs.
    Uses pdfplumber for native text & table layout extraction,
    falling back to PyMuPDF + RapidOCR when extracted text is < 30 characters.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.ocr_engine = OCREngine()

    def parse(self) -> Dict[int, List[Dict[str, Any]]]:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")

        parsed_data = {}
        with pdfplumber.open(self.file_path) as pdf:
            for idx, page in enumerate(pdf.pages):
                page_num = idx + 1
                elements = []
                
                # 1. Extract tables
                tables = page.extract_tables()
                table_objects = page.find_tables()
                table_bboxes = [t.bbox for t in table_objects]
                
                # 2. Extract non-table text
                page_text = ""
                if table_bboxes:
                    def not_within_table(obj):
                        if obj.get("object_type") == "char":
                            x0, top, x1, bottom = obj["x0"], obj["top"], obj["x1"], obj["bottom"]
                            for tx0, ttop, tx1, tbottom in table_bboxes:
                                if x0 >= tx0 - 2 and x1 <= tx1 + 2 and top >= ttop - 2 and bottom <= tbottom + 2:
                                    return False
                        return True
                    
                    non_table_page = page.filter(not_within_table)
                    page_text = non_table_page.extract_text() or ""
                else:
                    page_text = page.extract_text() or ""

                if not page_text.strip():
                    page_text = page.extract_text() or ""

                from app.ingestion.text_cleaner import TextCleaner
                cleaned_text_lines = []
                if page_text:
                    for line in page_text.splitlines():
                        line_cleaned = TextCleaner.clean_text(" ".join(line.split()))
                        if line_cleaned:
                            cleaned_text_lines.append(line_cleaned)

                # --- SCANNED PAGE OCR FALLBACK LAYER ---
                if len("\n".join(cleaned_text_lines).strip()) < 30 and not tables:
                    ocr_lines = self.ocr_engine.extract_text_from_pdf_page(self.file_path, page_num)
                    if ocr_lines:
                        cleaned_text_lines = ocr_lines

                if cleaned_text_lines:
                    elements.append({
                        "type": "text",
                        "text": "\n".join(cleaned_text_lines)
                    })
                    
                if tables:
                    for tbl in tables:
                        headers = [str(cell).strip() if cell else "" for cell in tbl[0]] if tbl else []
                        rows = [[str(cell).strip() if cell else "" for cell in row] for row in tbl[1:]] if len(tbl) > 1 else []
                        elements.append({
                            "type": "table",
                            "headers": headers,
                            "rows": rows,
                            "text": self._format_table_markdown(headers, rows)
                        })

                parsed_data[page_num] = elements

        return parsed_data

    def _format_table_markdown(self, headers: List[str], rows: List[List[str]]) -> str:
        if not headers and not rows:
            return ""
        valid_cols = [i for i, h in enumerate(headers) if h != ""]
        if not valid_cols:
            valid_cols = list(range(min(10, max(len(r) for r in rows) if rows else 0)))

        clean_headers = [headers[i] for i in valid_cols if i < len(headers)]
        lines = []
        header_line = "| " + " | ".join(clean_headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(clean_headers)) + " |"
        lines.append(header_line)
        lines.append(sep_line)
        for r in rows:
            r_cells = [r[i] if i < len(r) else "" for i in valid_cols]
            lines.append("| " + " | ".join(r_cells) + " |")
        return "\n".join(lines)
