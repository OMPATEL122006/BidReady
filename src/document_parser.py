import os
import sys
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extracts text page-by-page from a PDF file.
    
    Args:
        pdf_path (str): The absolute or relative path to the PDF file.
        
    Returns:
        dict: A dictionary mapping page numbers (1-indexed) to their extracted text.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    reader = PdfReader(pdf_path)
    extracted_data = {}
    
    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        text = page.extract_text()
        if text and text.strip():
            # Clean up whitespace on each line, but preserve newlines to protect tables and forms
            lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
            extracted_data[page_num] = "\n".join(lines)
            
    return extracted_data
