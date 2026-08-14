import json
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

def create_chunks(extracted_data: dict, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list:
    """
    Splits page-by-page extracted text into overlapping chunks with metadata.
    Compatibility wrapper returning default text chunks with structured headers.
    """
    all_chunks = []
    chunk_index = 0
    
    for page_num, text in extracted_data.items():
        lines = text.split("\n")
        current_chunk_lines = []
        current_length = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if len(line) > chunk_size:
                if current_chunk_lines:
                    chunk_text = " ".join(current_chunk_lines)
                    all_chunks.append({
                        "chunk_id": chunk_index,
                        "text": chunk_text,
                        "metadata": {
                            "page_number": page_num,
                            "char_start": 0,
                            "char_end": len(chunk_text),
                            "length": len(chunk_text),
                            "chunk_type": "text",
                            "structured_json": "{}"
                        }
                    })
                    chunk_index += 1
                    current_chunk_lines = []
                    current_length = 0
                
                start = 0
                while start < len(line):
                    end = min(start + chunk_size, len(line))
                    chunk_text = line[start:end]
                    all_chunks.append({
                        "chunk_id": chunk_index,
                        "text": chunk_text,
                        "metadata": {
                            "page_number": page_num,
                            "char_start": start,
                            "char_end": end,
                            "length": len(chunk_text),
                            "chunk_type": "text",
                            "structured_json": "{}"
                        }
                    })
                    chunk_index += 1
                    start += (chunk_size - overlap)
                continue

            if current_length + len(line) > chunk_size and current_chunk_lines:
                chunk_text = " ".join(current_chunk_lines)
                all_chunks.append({
                    "chunk_id": chunk_index,
                    "text": chunk_text,
                    "metadata": {
                        "page_number": page_num,
                        "char_start": 0,
                        "char_end": len(chunk_text),
                        "length": len(chunk_text),
                        "chunk_type": "text",
                        "structured_json": "{}"
                    }
                })
                chunk_index += 1
                
                overlap_lines = current_chunk_lines[-2:] if len(current_chunk_lines) >= 2 else current_chunk_lines[-1:]
                current_chunk_lines = list(overlap_lines)
                current_length = sum(len(l) for l in current_chunk_lines) + len(current_chunk_lines) - 1
            
            current_chunk_lines.append(line)
            current_length += len(line) + 1
            
        if current_chunk_lines:
            chunk_text = " ".join(current_chunk_lines)
            all_chunks.append({
                "chunk_id": chunk_index,
                "text": chunk_text,
                "metadata": {
                    "page_number": page_num,
                    "char_start": 0,
                    "char_end": len(chunk_text),
                    "length": len(chunk_text),
                    "chunk_type": "text",
                    "structured_json": "{}"
                }
            })
            chunk_index += 1
            
    return all_chunks

def chunk_document(parsed_data: dict, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list:
    """
    Chunks a parsed document consisting of structural page elements.
    
    Returns:
        list: A list of dicts. Each dict contains:
            - "chunk_id" (int)
            - "text" (str)
            - "metadata" (dict) containing:
                - "page_number" (int)
                - "chunk_type" (str: "text" | "table" | "boq")
                - "structured_json" (str: JSON string of structured fields)
    """
    # Check if input is flat strings mapping (compatibility fallback)
    first_val = next(iter(parsed_data.values())) if parsed_data else None
    if isinstance(first_val, str):
        return create_chunks(parsed_data, chunk_size, overlap)
        
    all_chunks = []
    chunk_index = 0
    
    for page_num, elements in parsed_data.items():
        for el in elements:
            el_type = el["type"]
            el_text = el["text"]
            
            if el_type == "text":
                lines = el_text.split("\n")
                current_chunk_lines = []
                current_length = 0
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    if len(line) > chunk_size:
                        if current_chunk_lines:
                            chunk_text = " ".join(current_chunk_lines)
                            all_chunks.append({
                                "chunk_id": chunk_index,
                                "text": chunk_text,
                                "metadata": {
                                    "page_number": page_num,
                                    "char_start": 0,
                                    "char_end": len(chunk_text),
                                    "length": len(chunk_text),
                                    "chunk_type": "text",
                                    "structured_json": "{}"
                                }
                            })
                            chunk_index += 1
                            current_chunk_lines = []
                            current_length = 0
                        
                        start = 0
                        while start < len(line):
                            end = min(start + chunk_size, len(line))
                            chunk_text = line[start:end]
                            all_chunks.append({
                                "chunk_id": chunk_index,
                                "text": chunk_text,
                                "metadata": {
                                    "page_number": page_num,
                                    "char_start": start,
                                    "char_end": end,
                                    "length": len(chunk_text),
                                    "chunk_type": "text",
                                    "structured_json": "{}"
                                }
                            })
                            chunk_index += 1
                            start += (chunk_size - overlap)
                        continue
                        
                    if current_length + len(line) > chunk_size and current_chunk_lines:
                        chunk_text = " ".join(current_chunk_lines)
                        all_chunks.append({
                            "chunk_id": chunk_index,
                            "text": chunk_text,
                            "metadata": {
                                "page_number": page_num,
                                "char_start": 0,
                                "char_end": len(chunk_text),
                                "length": len(chunk_text),
                                "chunk_type": "text",
                                "structured_json": "{}"
                            }
                        })
                        chunk_index += 1
                        
                        overlap_lines = current_chunk_lines[-2:] if len(current_chunk_lines) >= 2 else current_chunk_lines[-1:]
                        current_chunk_lines = list(overlap_lines)
                        current_length = sum(len(l) for l in current_chunk_lines) + len(current_chunk_lines) - 1
                        
                    current_chunk_lines.append(line)
                    current_length += len(line) + 1
                    
                if current_chunk_lines:
                    chunk_text = " ".join(current_chunk_lines)
                    all_chunks.append({
                        "chunk_id": chunk_index,
                        "text": chunk_text,
                        "metadata": {
                            "page_number": page_num,
                            "char_start": 0,
                            "char_end": len(chunk_text),
                            "length": len(chunk_text),
                            "chunk_type": "text",
                            "structured_json": "{}"
                        }
                    })
                    chunk_index += 1
                    
            elif el_type == "table":
                all_chunks.append({
                    "chunk_id": chunk_index,
                    "text": el_text,
                    "metadata": {
                        "page_number": page_num,
                        "char_start": 0,
                        "char_end": len(el_text),
                        "length": len(el_text),
                        "chunk_type": "table",
                        "structured_json": json.dumps({"headers": el.get("headers", []), "rows": el.get("rows", [])})
                    }
                })
                chunk_index += 1
                
            elif el_type == "boq":
                rows = el.get("rows", [])
                for row in rows:
                    row_text = f"BOQ Item {row['item_no']}: {row['description']} | Qty: {row['quantity']} {row['unit']} | Rate: {row['rate']} | Amount: {row['amount']}"
                    all_chunks.append({
                        "chunk_id": chunk_index,
                        "text": row_text,
                        "metadata": {
                            "page_number": page_num,
                            "char_start": 0,
                            "char_end": len(row_text),
                            "length": len(row_text),
                            "chunk_type": "boq",
                            "structured_json": json.dumps(row)
                        }
                    })
                    chunk_index += 1
                    
                # Also yield the complete Markdown representation of the BOQ table as a table
                all_chunks.append({
                    "chunk_id": chunk_index,
                    "text": el_text,
                    "metadata": {
                        "page_number": page_num,
                        "char_start": 0,
                        "char_end": len(el_text),
                        "length": len(el_text),
                        "chunk_type": "table",
                        "structured_json": json.dumps({"headers": ["Item", "Description", "Qty", "Unit", "Rate", "Amount"], "rows": [[r["item_no"], r["description"], str(r["quantity"]), r["unit"], str(r["rate"]), str(r["amount"])] for r in rows]})
                    }
                })
                chunk_index += 1
                
    return all_chunks
