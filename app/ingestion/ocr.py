import os
from typing import List, Optional
from app.config.logging import logger

class OCREngine:
    """
    Wrapper for PyMuPDF page rendering and RapidOCR text extraction.
    """
    def __init__(self):
        self._engine = None

    def _init_engine(self):
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._engine = RapidOCR()
            except Exception as e:
                logger.error(f"Failed to initialize RapidOCR engine: {e}")
                raise e

    def extract_text_from_pdf_page(self, pdf_path: str, page_number: int, dpi: int = 200) -> List[str]:
        """
        Renders a PDF page as pixmap image using PyMuPDF and runs RapidOCR on it.
        """
        self._init_engine()
        try:
            import pymupdf
            doc = pymupdf.open(pdf_path)
            if page_number < 1 or page_number > len(doc):
                return []
                
            fitz_page = doc[page_number - 1]
            pix = fitz_page.get_pixmap(dpi=dpi)
            
            temp_img_path = f"scratch_ocr_p{page_number}_{os.path.basename(pdf_path)}.png"
            pix.save(temp_img_path)
            
            ocr_res, _ = self._engine(temp_img_path)
            
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
                
            if ocr_res:
                from app.ingestion.text_cleaner import TextCleaner
                lines = [TextCleaner.clean_text(line[1].strip()) for line in ocr_res if line[1] and line[1].strip()]
                return lines
        except Exception as e:
            logger.warning(f"OCR execution warning on {pdf_path} Page {page_number}: {e}")
            
        return []
