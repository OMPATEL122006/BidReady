import os
import re
import json

# We will lazily import pdfplumber inside the classes to ensure 
# the environment has time to load the library.

class TenderParser:
    def __init__(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        self.pdf_path = pdf_path

    def parse(self) -> dict:
        """
        Parses the PDF and returns a dict mapping page numbers (1-indexed)
        to a list of structural element dicts:
        [
            {"type": "text", "text": str},
            {"type": "table", "text": str, "headers": list, "rows": list},
            {"type": "boq", "text": str, "rows": list of dict}
        ]
        """
        import pdfplumber
        
        parsed_data = {}
        with pdfplumber.open(self.pdf_path) as pdf:
            for idx, page in enumerate(pdf.pages):
                page_num = idx + 1
                elements = []
                
                # 1. Extract tables
                tables = page.extract_tables()
                table_objects = page.find_tables()
                table_bboxes = [t.bbox for t in table_objects]
                
                # 2. Extract non-table text by filtering characters within table boundaries
                page_text = ""
                if table_bboxes:
                    def not_within_table(obj):
                        if obj.get("object_type") == "char":
                            x0, top, x1, bottom = obj["x0"], obj["top"], obj["x1"], obj["bottom"]
                            for tx0, ttop, tx1, tbottom in table_bboxes:
                                # Safe boundary margin of 2 units
                                if x0 >= tx0 - 2 and x1 <= tx1 + 2 and top >= ttop - 2 and bottom <= tbottom + 2:
                                    return False
                        return True
                    
                    non_table_page = page.filter(not_within_table)
                    page_text = non_table_page.extract_text() or ""
                else:
                    page_text = page.extract_text() or ""

                # Extract and clean text lines
                cleaned_text_lines = []
                if page_text:
                    for line in page_text.splitlines():
                        line_cleaned = " ".join(line.split())
                        if line_cleaned:
                            cleaned_text_lines.append(line_cleaned)
                            
                if cleaned_text_lines:
                    elements.append({
                        "type": "text",
                        "text": "\n".join(cleaned_text_lines)
                    })

                # Format extracted tables
                for t in tables:
                    if not t or not any(any(cell for cell in row) for row in t):
                        continue
                    
                    cleaned_rows = []
                    for row in t:
                        cleaned_row = []
                        for cell in row:
                            cleaned_row.append(str(cell or "").strip().replace("\n", " "))
                        if any(cell for cell in cleaned_row):
                            cleaned_rows.append(cleaned_row)
                            
                    if not cleaned_rows:
                        continue
                        
                    headers = cleaned_rows[0]
                    data_rows = cleaned_rows[1:]
                    
                    # Detect if BOQ structure matches
                    is_boq = False
                    boq_indices = {}
                    norm_headers = [h.lower() for h in headers]
                    
                    description_keywords = ["description", "particulars", "name of work", "specification", "item of work"]
                    qty_keywords = ["qty", "quantity", "quantities"]
                    rate_keywords = ["rate", "unit rate", "cost per"]
                    amount_keywords = ["amount", "estimated amount", "total amount", "cost", "estimated cost"]
                    unit_keywords = ["unit", "per"]
                    item_keywords = ["item", "sl", "no", "number", "code"]

                    for c_idx, h in enumerate(norm_headers):
                        if any(k in h for k in description_keywords):
                            boq_indices["description"] = c_idx
                        elif any(k in h for k in qty_keywords):
                            boq_indices["quantity"] = c_idx
                        elif any(k in h for k in rate_keywords):
                            boq_indices["rate"] = c_idx
                        elif any(k in h for k in amount_keywords):
                            boq_indices["amount"] = c_idx
                        elif any(k in h for k in unit_keywords):
                            boq_indices["unit"] = c_idx
                        elif any(k in h for k in item_keywords):
                            boq_indices["item_no"] = c_idx

                    # Classify as BOQ if it has description and at least two of (quantity, rate, amount, item_no)
                    # AND is not a general NIT summary table (doesn't contain "emd", "earnest", or "opening" in headers)
                    if "description" in boq_indices and sum(1 for k in ["quantity", "rate", "amount", "item_no"] if k in boq_indices) >= 2:
                        is_nit_summary = any(any(w in h for w in ["emd", "earnest", "opening"]) for h in norm_headers)
                        if not is_nit_summary:
                            is_boq = True

                    # Convert table to Markdown format
                    markdown_lines = []
                    col_widths = [max(len(row[i]) for row in cleaned_rows) for i in range(len(headers))]
                    col_widths = [max(w, 3) for w in col_widths]
                    
                    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
                    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
                    markdown_lines.append(header_line)
                    markdown_lines.append(sep_line)
                    
                    for row in data_rows:
                        # Handle row length mismatch safely
                        row_cells = row + [""] * (len(headers) - len(row))
                        row_line = "| " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row_cells[:len(headers)])) + " |"
                        markdown_lines.append(row_line)
                        
                    markdown_table = "\n".join(markdown_lines)

                    if is_boq:
                        structured_boq_rows = []
                        for row in data_rows:
                            item_val = row[boq_indices["item_no"]].strip() if "item_no" in boq_indices and boq_indices["item_no"] < len(row) else ""
                            desc_val = row[boq_indices["description"]].strip() if "description" in boq_indices and boq_indices["description"] < len(row) else ""
                            qty_val = row[boq_indices["quantity"]].strip() if "quantity" in boq_indices and boq_indices["quantity"] < len(row) else ""
                            unit_val = row[boq_indices["unit"]].strip() if "unit" in boq_indices and boq_indices["unit"] < len(row) else ""
                            rate_val = row[boq_indices["rate"]].strip() if "rate" in boq_indices and boq_indices["rate"] < len(row) else ""
                            amount_val = row[boq_indices["amount"]].strip() if "amount" in boq_indices and boq_indices["amount"] < len(row) else ""
                            
                            item_match = re.match(r'^(\d+(?:\.\d+)*)', item_val)
                            clean_item_no = item_match.group(1) if item_match else item_val

                            def parse_number(val):
                                val_cleaned = re.sub(r'[^\d.]', '', val)
                                try:
                                    return float(val_cleaned) if val_cleaned else 0.0
                                except ValueError:
                                    return 0.0

                            structured_row = {
                                "item_no": clean_item_no,
                                "description": desc_val,
                                "quantity": parse_number(qty_val),
                                "unit": unit_val,
                                "rate": parse_number(rate_val),
                                "amount": parse_number(amount_val)
                            }
                            if desc_val:
                                structured_boq_rows.append(structured_row)

                        elements.append({
                            "type": "boq",
                            "text": markdown_table,
                            "rows": structured_boq_rows
                        })
                    else:
                        elements.append({
                            "type": "table",
                            "text": markdown_table,
                            "headers": headers,
                            "rows": data_rows
                        })
                        
                parsed_data[page_num] = elements
                
        return parsed_data

def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Compatibility wrapper returning plain text page-by-page mapping,
    converting tables into inline Markdown tables.
    """
    parser = TenderParser(pdf_path)
    parsed_pages = parser.parse()
    
    flat_data = {}
    for page_num, elements in parsed_pages.items():
        page_texts = []
        for el in elements:
            page_texts.append(el["text"])
        flat_data[page_num] = "\n\n".join(page_texts)
        
    return flat_data
