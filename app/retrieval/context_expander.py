from typing import List, Dict, Any

class ContextExpander:
    """
    Stitches adjacent chunks on the same page when a clause or list is cut off mid-sentence.
    """
    @staticmethod
    def expand_candidate_contexts(cand_list: List[Dict[str, Any]], chunks_cache: List[Dict[str, Any]]):
        if not cand_list or not chunks_cache:
            return

        continuation_markers = (':', ',', ';', '-', '—')
        continuation_words = {"following", "under", "below", "include", "contains", "forms", "schedule", "clause"}

        for cand in cand_list:
            text = cand["text"].strip()
            if not text:
                continue

            needs_expansion = False
            last_char = text[-1]

            if last_char in continuation_markers:
                needs_expansion = True

            last_words = text.lower().split()[-3:]
            if any(cw in lw for cw in continuation_words for lw in last_words):
                needs_expansion = True

            if last_char.isalnum() and last_char not in ('.', '!', '?'):
                needs_expansion = True

            if needs_expansion:
                c_idx = cand.get("chunk_id", 0)
                next_idx = c_idx + 1
                if next_idx < len(chunks_cache):
                    next_chunk = chunks_cache[next_idx]
                    if next_chunk["metadata"]["page_number"] == cand["metadata"]["page_number"]:
                        cand["text"] = cand["text"] + "\n[CONTINUATION]: " + next_chunk["text"]
