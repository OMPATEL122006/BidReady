from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

def create_chunks(extracted_data: dict, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list:
    """
    Splits page-by-page extracted text into overlapping chunks with metadata.
    
    Args:
        extracted_data (dict): Dictionary mapping page numbers to text.
        chunk_size (int): Max number of characters per chunk.
        overlap (int): Number of characters to overlap between consecutive chunks.
        
    Returns:
        list: A list of dictionaries representing chunks. Each chunk contains:
              - "chunk_id" (int)
              - "text" (str)
              - "metadata" (dict) containing page_number, char_start, char_end, length.
    """
    all_chunks = []
    chunk_index = 0
    
    for page_num, text in extracted_data.items():
        # Split by newlines to respect lines and tables
        lines = text.split("\n")
        
        current_chunk_lines = []
        current_length = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # If a single line is extremely long, split it by sentences or characters
            if len(line) > chunk_size:
                # If we have existing lines in our buffer, yield them first
                if current_chunk_lines:
                    chunk_text = " ".join(current_chunk_lines)
                    all_chunks.append({
                        "chunk_id": chunk_index,
                        "text": chunk_text,
                        "metadata": {
                            "page_number": page_num,
                            "char_start": 0,
                            "char_end": len(chunk_text),
                            "length": len(chunk_text)
                        }
                    })
                    chunk_index += 1
                    current_chunk_lines = []
                    current_length = 0
                
                # Split this long line by characters
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
                            "length": len(chunk_text)
                        }
                    })
                    chunk_index += 1
                    start += (chunk_size - overlap)
                continue

            # If adding this line exceeds the chunk size, yield the current buffer
            if current_length + len(line) > chunk_size and current_chunk_lines:
                chunk_text = " ".join(current_chunk_lines)
                all_chunks.append({
                    "chunk_id": chunk_index,
                    "text": chunk_text,
                    "metadata": {
                        "page_number": page_num,
                        "char_start": 0,
                        "char_end": len(chunk_text),
                        "length": len(chunk_text)
                    }
                })
                chunk_index += 1
                
                # Overlap: Keep the last 1 or 2 lines for context continuity
                overlap_lines = current_chunk_lines[-2:] if len(current_chunk_lines) >= 2 else current_chunk_lines[-1:]
                current_chunk_lines = list(overlap_lines)
                current_length = sum(len(l) for l in current_chunk_lines) + len(current_chunk_lines) - 1
            
            current_chunk_lines.append(line)
            current_length += len(line) + 1 # +1 for separating space
            
        # Append the final remaining lines on the page
        if current_chunk_lines:
            chunk_text = " ".join(current_chunk_lines)
            all_chunks.append({
                "chunk_id": chunk_index,
                "text": chunk_text,
                "metadata": {
                    "page_number": page_num,
                    "char_start": 0,
                    "char_end": len(chunk_text),
                    "length": len(chunk_text)
                }
            })
            chunk_index += 1
            
    return all_chunks
