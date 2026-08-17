import re
import json
from typing import List, Dict, Any
from app.config.settings import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from app.models.chunk import Chunk, ChunkType

class TextChunker:
    """
    Structure-aware chunker for tender documents.
    Preserves document_id, page_number, section, and clause metadata.
    """
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self,
        parsed_data: Dict[int, List[Dict[str, Any]]],
        source_doc: str = "Unknown",
        document_id: str = "doc",
        tender_id: str = "default_tender",
        document_type: str = "OTHER",
        document_version: int = 1
    ) -> List[Chunk]:
        chunks = []
        global_chunk_id = 0

        for page_num in sorted(parsed_data.keys()):
            elements = parsed_data[page_num]

            for el in elements:
                el_type = el.get("type", "text")

                if el_type == "boq":
                    structured_rows = el.get("rows", [])
                    sheet_name = el.get("sheet_name", "BoQ")
                    markdown_table = el.get("text", "")

                    if structured_rows:
                        for row_idx, row_data in enumerate(structured_rows, 1):
                            global_chunk_id += 1
                            item_no = row_data.get("item_no", "")
                            desc = row_data.get("description", "")
                            qty = row_data.get("quantity", 0.0)
                            unit = row_data.get("unit", "")
                            rate = row_data.get("rate", 0.0)
                            amt = row_data.get("amount", 0.0)

                            item_text = (
                                f"BOQ Item {item_no}: {desc} | "
                                f"Qty: {qty} {unit} | Rate: {rate} | Amount: {amt}"
                            )

                            c = Chunk(
                                chunk_id=global_chunk_id,
                                text=item_text,
                                chunk_type=ChunkType.BOQ,
                                page_number=page_num,
                                char_start=0,
                                char_end=len(item_text),
                                document_id=document_id,
                                source_doc=source_doc,
                                tender_id=tender_id,
                                document_type=document_type,
                                document_version=document_version,
                                content_type="boq",
                                sheet=sheet_name,
                                row=row_idx,
                                structured_json=json.dumps(row_data)
                            )
                            chunks.append(c)
                    else:
                        global_chunk_id += 1
                        c = Chunk(
                            chunk_id=global_chunk_id,
                            text=markdown_table,
                            chunk_type=ChunkType.BOQ,
                            page_number=page_num,
                            char_start=0,
                            char_end=len(markdown_table),
                            document_id=document_id,
                            source_doc=source_doc,
                            tender_id=tender_id,
                            document_type=document_type,
                            document_version=document_version,
                            content_type="boq",
                            sheet=sheet_name,
                            structured_json="{}"
                        )
                        chunks.append(c)

                elif el_type == "table":
                    tbl_text = el.get("text", "")
                    if tbl_text.strip():
                        global_chunk_id += 1
                        c = Chunk(
                            chunk_id=global_chunk_id,
                            text=tbl_text,
                            chunk_type=ChunkType.TABLE,
                            page_number=page_num,
                            char_start=0,
                            char_end=len(tbl_text),
                            document_id=document_id,
                            source_doc=source_doc,
                            tender_id=tender_id,
                            document_type=document_type,
                            document_version=document_version,
                            content_type="table",
                            structured_json="{}"
                        )
                        chunks.append(c)

                else:
                    text = el.get("text", "")
                    if not text.strip():
                        continue

                    page_text_chunks = self._chunk_text_with_boundaries(text)
                    for char_start, char_end, c_text in page_text_chunks:
                        global_chunk_id += 1
                        c = Chunk(
                            chunk_id=global_chunk_id,
                            text=c_text,
                            chunk_type=ChunkType.TEXT,
                            page_number=page_num,
                            char_start=char_start,
                            char_end=char_end,
                            document_id=document_id,
                            source_doc=source_doc,
                            tender_id=tender_id,
                            document_type=document_type,
                            document_version=document_version,
                            content_type="text",
                            structured_json="{}"
                        )
                        chunks.append(c)

        return chunks

    def _chunk_text_with_boundaries(self, text: str) -> List[tuple]:
        if len(text) <= self.chunk_size:
            return [(0, len(text), text)]

        chunks = []
        sentences = re.split(r'(?<=[.!?\n])\s+', text)

        current_chunk = []
        current_len = 0
        chunk_start_idx = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current_len + sentence_len > self.chunk_size and current_chunk:
                chunk_str = " ".join(current_chunk)
                chunk_end_idx = chunk_start_idx + len(chunk_str)
                chunks.append((chunk_start_idx, chunk_end_idx, chunk_str))

                overlap_words = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap_words.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break

                current_chunk = overlap_words
                current_len = overlap_len
                chunk_start_idx = max(0, chunk_end_idx - overlap_len)

            current_chunk.append(sentence)
            current_len += sentence_len

        if current_chunk:
            chunk_str = " ".join(current_chunk)
            chunk_end_idx = chunk_start_idx + len(chunk_str)
            chunks.append((chunk_start_idx, chunk_end_idx, chunk_str))

        return chunks
