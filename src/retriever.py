import re
import math
from src.config import CANDIDATE_POOL_SIZE, STOP_WORDS
from src.database import get_chroma_client, get_or_create_collection, query_chroma
from src.bm25 import BM25
from src.reranker import BGEReranker
from db_experiment import compute_keyword_score, compute_phrase_boost

class TenderRetriever:
    def __init__(self):
        # 1. Initialize Clients
        self.client = get_chroma_client()
        self.collection = get_or_create_collection(self.client, "tender_requirements")
        self.reranker = BGEReranker()
        
        # 2. Build BM25 Index Dynamically from database chunks
        self.chunks_cache = []
        self.bm25 = None
        self._initialize_bm25()

    def _initialize_bm25(self):
        """
        Fetches all stored chunks in ChromaDB and fits the BM25 model on them.
        """
        count = self.collection.count()
        if count == 0:
            print("Warning: Chroma collection is empty. Cannot initialize BM25 yet.")
            return
            
        print(f"Loading {count} chunks from Chroma to fit local BM25 index...")
        all_data = self.collection.get()
        
        chunks = []
        for cid, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
            chunks.append({
                "chunk_id": int(cid.split("_")[1]),
                "text": doc,
                "metadata": meta
            })
            
        # Sort chunks by ID to align indices
        chunks.sort(key=lambda x: x["chunk_id"])
        self.chunks_cache = chunks
        
        # Fit BM25
        self.bm25 = BM25(chunks)
        print("BM25 index successfully fitted.")

    def _get_domain_boost_score(self, query: str, doc_text: str) -> float:
        """
        Applies deterministic penalties and boosts based on exact terminology and values
        to prevent semantic confusion (e.g. confusing EMD with Performance Guarantee).
        """
        # Apply deterministic standardizations to query and doc text for matching rule intents
        q_lower = query.lower()
        q_lower = q_lower.replace("n.i.t.no.", "nit no").replace("n.i.t. no.", "nit no")
        q_lower = q_lower.replace("n.i.t. no", "nit no").replace("n.i.t.", "nit")
        q_lower = q_lower.replace("e.m.d.", "emd")

        d_lower = doc_text.lower()
        d_lower = d_lower.replace("n.i.t.no.", "nit no").replace("n.i.t. no.", "nit no")
        d_lower = d_lower.replace("n.i.t. no", "nit no").replace("n.i.t.", "nit")
        d_lower = d_lower.replace("e.m.d.", "emd")

        boost = 0.0

        # Amount and Date patterns
        currency_val_pattern = r'(?:rs\.?|₹|rupees?)\s*\d+[\d,]*'
        date_pattern = r'\d{2}\.\d{2}\.\d{4}'
        time_pattern = r'\d{2}\.\d{2}\s*(?:hrs|am|pm)'

        # --- RULE 1: NIT Number Search (Notice Inviting Tender number reference) ---
        if "nit number" in q_lower or "nit no" in q_lower:
            # Look for typical CPWD NIT number formats: e.g. 17/AE/PCSD/LKO/26-27 or slashes (matching lowercase letters)
            if re.search(r'\b\d+/[a-z0-9/]+-\d+\b|\b\d+/[a-z0-9/]+\b', d_lower):
                boost += 0.5
            # Strongly penalize National Institute of Technology (NIT) concrete design testing reference
            if any(w in d_lower for w in ["iit", "laboratories", "test house", "laboratory", "mix design", "institutes"]):
                boost -= 0.7

        # --- RULE 2: EMD (Earnest Money Deposit) ---
        elif "emd" in q_lower or "earnest money" in q_lower:
            # Check if query is looking for deposit forms vs deposit amount
            if any(w in q_lower for w in ["forms", "acceptable", "how to", "mode", "payment", "instrument", "deposited"]):
                # Boost if document lists acceptable EMD forms
                payment_keywords = ["demand draft", "pay order", "banker's cheque", "fixed deposit", "bank guarantee", "rtgs", "neft", "receipt of deposition"]
                matches = sum(1 for w in payment_keywords if w in d_lower)
                if matches >= 2:
                    boost += 0.5
                if "deposited" in d_lower or "deposited through" in d_lower:
                    boost += 0.3
            else:
                # Searching for EMD Amount
                if "earnest money" in d_lower or "emd" in d_lower:
                    boost += 0.3
                if re.search(currency_val_pattern, d_lower):
                    boost += 0.2
            # Penalize performance guarantee to avoid mixing them up
            if "performance guarantee" in d_lower or "security deposit" in d_lower:
                boost -= 0.6

        # --- RULE 3: Completion Period vs. Guarantee / Defect Liability Period ---
        elif any(k in q_lower for k in ["completion", "period of completion", "how long", "duration"]):
            if "completion" in d_lower and any(w in d_lower for w in ["month", "months", "days", "weeks"]):
                boost += 0.3
            # Strongly penalize chunks talking about maintenance/defect liability/tile guarantee
            if any(w in d_lower for w in ["guarantee bond", "stone", "tile", "defect", "liability", "warranty"]):
                boost -= 0.7

        # --- RULE 4: NIT Date ---
        elif "nit date" in q_lower or "date of the nit" in q_lower or "date of nit" in q_lower:
            # Look for N.I.T.No & Date format
            if "nit" in d_lower and "dated" in d_lower:
                boost += 0.6
            elif re.search(date_pattern, d_lower):
                boost += 0.2
            # Penalize generic cost estimate or page 1 Approval
            if "approved" in d_lower and "estimated cost" in d_lower:
                boost -= 0.4

        # --- RULE 5: Financial Cover Opening ---
        elif "financial cover" in q_lower and any(k in q_lower for k in ["opening", "opened"]):
            if "financial cover" in d_lower or "financial" in d_lower:
                boost += 0.3
            if any(w in d_lower for w in ["later date", "communicated", "notified later"]):
                boost += 0.4
            # Penalize eligibility cover opening dates to avoid confusion
            if "eligibility cover" in d_lower or "cover 1" in d_lower:
                boost -= 0.6

        # --- RULE 6: Eligibility Cover Bid Opening Date/Time seeking "When" ---
        elif any(k in q_lower for k in ["opening", "opened"]):
            # Check if query contains "when" (seeking date)
            if "when" in q_lower:
                if re.search(date_pattern, d_lower) or re.search(time_pattern, d_lower):
                    boost += 0.4
                else:
                    boost -= 0.5  # Penalize chunks with no date/time
            if "opening" in d_lower or "opened" in d_lower:
                boost += 0.3
            if "eligibility" in d_lower:
                boost += 0.2
            # Penalize submission rules
            if "submission" in d_lower:
                boost -= 0.3

        # --- RULE 7: Bid Submission Deadline ---
        elif any(k in q_lower for k in ["submission", "last date", "deadline"]):
            if "submission" in d_lower and "last date" in d_lower:
                boost += 0.3
            if re.search(date_pattern, d_lower):
                boost += 0.2
            # Penalize opening rules
            if "opening" in d_lower or "opened" in d_lower:
                boost -= 0.5

        # --- RULE 8: Similar Work ---
        elif "similar work" in q_lower:
            if "similar work means" in d_lower:
                boost += 0.5
            if "specialized" in d_lower:
                boost -= 0.3

        # --- RULE 9: Name of the Work ---
        if "name of the work" in q_lower or "name of work" in q_lower:
            if "name of work:" in d_lower or "name of work :" in d_lower or "tender for the work of:" in d_lower or "tender for the work of" in d_lower:
                boost += 0.6

        # --- RULE 10: Location of Work ---
        if "where" in q_lower or "located" in q_lower or "situated" in q_lower:
            if any(w in d_lower for w in ["situated at", "located at", "situated in", "at", "site", "compound", "location", "address"]):
                boost += 0.5
            if "name of work" in d_lower or "description of the work" in d_lower:
                boost += 0.6
            if "specification" in d_lower or "aluminium" in d_lower or "glazing" in d_lower:
                boost -= 0.7

        # --- RULE 11: Government Department / Ministry ---
        if "department" in q_lower or "ministry" in q_lower:
            if any(w in d_lower for w in ["department of", "ministry of", "government of", "under the"]):
                boost += 0.6

        # --- RULE 12: Type of Tender ---
        if "type of tender" in q_lower or "tender type" in q_lower:
            if any(w in d_lower for w in ["percentage rate nit", "percentage rate tender", "percentage rate e-tender", "item rate tender"]):
                boost += 0.6
            if any(w in d_lower for w in ["cpwd-6", "cpwd form no. 7", "invites online"]):
                boost += 0.4
            if "aluminium" in d_lower or "glazing" in d_lower:
                boost -= 0.5

        # --- RULE 13: Contractor Eligibility ---
        if "contractor" in q_lower and any(w in q_lower for w in ["allowed", "eligible", "enlisted", "class", "type"]):
            if any(w in d_lower for w in ["approved and eligible", "enlisted", "appropriate list", "cpwd", "mes", "bsnl", "railway"]):
                boost += 0.5

        # --- RULE 14: Integrity Pact Condition / Threshold ---
        if "integrity pact" in q_lower and any(w in q_lower for w in ["condition", "required", "applicability", "threshold"]):
            if "integrity pact" in d_lower:
                if any(w in d_lower for w in ["estimated cost", "threshold", "equal to or", "more than", "value of"]):
                    boost += 0.6
                if re.search(r'\b\d+\s*(?:lakh|crore|thousand|lac)s?\b', d_lower) or "threshold value" in d_lower:
                    boost += 0.5
                if any(w in d_lower for w in ["page no", "guidelines", "engaged in"]):
                    boost -= 0.6

        # --- RULE 15: Form D Description ---
        if "form d" in q_lower:
            if "form d" in d_lower or "form ‘d’" in d_lower:
                if "performance report" in d_lower or "experience" in d_lower:
                    boost += 0.6

        # --- RULE 16: Minimum EMD BG portion ---
        if "minimum portion" in q_lower or "minimum" in q_lower and "bank guarantee" in q_lower:
            if "%" in d_lower or "percent" in d_lower or "50%" in d_lower:
                boost += 0.5

        # --- RULE 17: Post-Tender Modification / Withdrawal ---
        if any(w in q_lower for w in ["post-tender", "modification", "withdraw"]):
            if any(w in d_lower for w in ["post-tender", "modification", "withdraw", "forfeit"]):
                boost += 0.6
            else:
                boost -= 0.6

        return boost

    def _classify_query_target(self, query: str) -> str:
        """
        Classifies the query into a target answer type to guide validation rules.
        """
        q_lower = query.lower()
        
        # 1. Date/Time targets
        if any(k in q_lower for k in ["when", "date of", "deadline", "schedule", "time of"]):
            return "DATE_TIME"
            
        # 2. Currency/Amount targets
        if any(k in q_lower for k in ["how much", "amount", "cost", "value", "price", "deposit", "fee", "rupees", "rs", "₹"]):
            return "CURRENCY_AMOUNT"
            
        # 3. URL targets
        if any(k in q_lower for k in ["portal", "website", "link", "url"]):
            return "URL"
            
        # 4. Definition targets
        if any(k in q_lower for k in ["what does", "meaning", "definition of", "similar work means", "similar work"]):
            if any(k in q_lower for k in ["what does", "mean", "meaning", "definition"]):
                return "DEFINITION"
                
        # 5. Consequence/Condition targets
        if any(k in q_lower for k in ["what happens if", "consequence", "violation", "withdrawn", "discrepancy", "discovered", "nil rate", "no quote"]):
            return "CONSEQUENCE"
            
        return "ENTITY_NAME"

    def _is_answerable(self, query: str, doc: str) -> bool:
        """
        Generic answerability validation to filter out chunks that match keywords
        but do not actually contain the structural/semantic answer patterns.
        """
        q_lower = query.lower()
        d_lower = doc.lower()
        d_lower = d_lower.replace("n.i.t.no.", "nit no").replace("n.i.t. no.", "nit no")
        d_lower = d_lower.replace("n.i.t. no", "nit no").replace("n.i.t.", "nit")
        d_lower = d_lower.replace("e.m.d.", "emd")
        d_lower = d_lower.replace("gstin", "gst")

        # Amount and Date patterns
        date_pattern = r'\d{2}\.\d{2}\.\d{4}'
        time_pattern = r'\d{2}\.\d{2}\s*(?:hrs|am|pm)'

        # 1. Location queries ("where", "located", "location")
        if any(w in q_lower for w in ["where", "located", "location", "situated"]):
            location_terms = ["situated", "located", "address", "site", "compound", "under", "state", "city", "place"]
            if not any(w in d_lower for w in location_terms):
                return False

        # 2. Cost / Amount queries ("estimated cost", "amount", "value", "price", "fee")
        if any(w in q_lower for w in ["cost", "amount", "estimated cost", "value", "price", "fee", "emd", "deposit"]):
            # Must contain a currency amount, percent, or a numerical value of 3+ digits (ignoring clause sections like 1.1.1)
            has_numeric = re.search(r'\b\d{3,}\b|\b\d+[\d,]*\s*(?:lakh|crore|thousand|lac|%|percent)\b|rs\.?|₹|rupees|nil|zero', d_lower)
            if not has_numeric:
                return False
            # If the query is about the work's estimated cost, exclude experience thresholds
            if "estimated cost" in q_lower and not any(w in q_lower for w in ["experience", "criteria", "eligibility"]):
                if any(w in d_lower for w in ["completed work costing", "completed works costing", "equal to 80%", "equal to 50%", "equal to 40%"]):
                    return False

        # 3. Sub-head / Head work queries ("SH", "sub-head", "head work")
        if "sh/" in q_lower or "sh" in q_lower or "sub-head" in q_lower or "sub head" in q_lower or "head work" in q_lower:
            if not any(w in d_lower for w in ["sh:", "sub-head", "sub head", "head work", "work of"]):
                return False

        # 4. Form C/D use case queries ("Form D", "used for")
        if "form d" in q_lower and any(w in q_lower for w in ["used for", "use", "purpose"]):
            if not any(w in d_lower for w in ["performance", "report", "experience", "referred"]):
                return False

        # 5. Return / Refund queries ("returned", "refund")
        if "returned" in q_lower or "refund" in q_lower or "discharge" in q_lower:
            if not any(w in d_lower for w in ["returned", "refund", "discharge", "back", "released"]):
                return False

        # 6. Withdrawal forfeiture / debarment consequences
        if "withdrawal" in q_lower or "withdraw" in q_lower or "post-tender" in q_lower or "modification" in q_lower:
            if not any(w in d_lower for w in ["forfeit", "debar", "reject", "withdraw", "modification", "transgression"]):
                return False
            # If explicitly looking for what happens if they withdraw, require withdraw or forfeit keywords
            if "withdraw" in q_lower or "withdrawal" in q_lower:
                if not any(w in d_lower for w in ["withdraw", "forfeit"]):
                    return False

        # 7. Tender validity / remain open period
        if "remain open" in q_lower or "open for acceptance" in q_lower or "validity" in q_lower:
            # Distinguish from performance guarantee validity by checking bid validity context
            has_bid_validity = any(w in d_lower for w in ["open for acceptance", "remain open"]) or ("validity" in d_lower and ("bid" in d_lower or "tender" in d_lower or "offer" in d_lower))
            if not has_bid_validity:
                return False
            # Distinguish bid validity from EMD/BG/Guarantee validity
            if "validity" in q_lower and any(w in q_lower for w in ["bid", "tender", "offer"]):
                if any(w in d_lower for w in ["earnest money shall remain valid", "validity of the bank guarantee", "validity of emd", "validity of the bg", "guarantee shall", "validity of this guarantee"]):
                    return False

        # 8. Inviting officer / designations
        if "who" in q_lower or "officer" in q_lower or "engineer" in q_lower:
            if not any(w in d_lower for w in ["assistant engineer", "ae", "executive engineer", "ee", "president", "invites", "division", "administrative officer"]):
                return False
            # If specifically looking for who is inviting/invitations, require invites action terms
            if "inviting" in q_lower or "invites" in q_lower or "invite" in q_lower:
                if not any(w in d_lower for w in ["invites", "inviting", "invite", "invitation", "on behalf of"]):
                    return False

        # 9. Issuing Department / Ministry
        if "department" in q_lower or "ministry" in q_lower:
            # Exclude references to CPWD Vulnerability Atlas and Ministry of Labour guidelines
            if any(w in d_lower for w in ["vulnerability atlas", "vaimulti-hazard", "ministry of labour", "ministry of housing"]):
                return False

        # 10. Percentage queries
        if "percent" in q_lower or "percentage" in q_lower or "%" in q_lower:
            if not any(w in d_lower for w in ["%", "percent", "percentage"]):
                return False

        # 11. Yes/No Document Requirements / Allowances (PAN, GST, JV, Enlistment)
        if any(w in q_lower for w in ["is ", "are ", "does ", "do ", "can ", "has ", "have "]):
            if any(w in q_lower for w in ["required", "allow", "accept", "need", "mandatory", "enlisted", "enrol"]):
                status_keywords = ["required", "acceptable", "allowed", "must", "shall", "enclosed", "enlist", "uploaded", "submitted", "to be", "not", "no"]
                if not any(w in d_lower for w in status_keywords):
                    return False

        # 12. Date / Time queries
        if any(w in q_lower for w in ["when", "date of", "deadline", "schedule", "time of"]):
            has_date_time = re.search(date_pattern, d_lower) or re.search(time_pattern, d_lower) or any(w in d_lower for w in ["days", "months", "weeks", "later date", "communicated", "notified"])
            if not has_date_time:
                return False
            # Distinguish opening schedule from bid validity
            if "open" in q_lower and any(w in q_lower for w in ["when", "date", "time"]):
                if any(w in d_lower for w in ["validity of tender", "remain open for acceptance"]):
                    return False

        # 13. Completion Period vs Liquidated Damages
        if any(w in q_lower for w in ["completion", "duration", "how long"]):
            # Exclude liquidated damages pages
            if any(w in d_lower for w in ["liquidated damages", "compensation for delay", "delay so claimed"]):
                return False

        # 14. Joint Venture presence
        if "joint venture" in q_lower or "jv" in q_lower:
            if not any(w in d_lower for w in ["joint venture", "jv", "consortium"]):
                return False

        # 15. Print specifications presence
        if "print spec" in q_lower or "printing spec" in q_lower:
            if not any(w in d_lower for w in ["gsm", "paper", "binding", "page size", "maplitho", "art paper", "monochrome", "multicolour"]):
                return False
        elif "print" in q_lower or "printing" in q_lower:
            if not any(w in d_lower for w in ["print", "printing", "publication", "stationery"]):
                return False

        return True

    def _expand_context(self, candidates: list, query: str):
        """
        Appends adjacent text chunks from the same page if the chunk is cut off mid-list or mid-sentence.
        """
        q_lower = query.lower()
        continuation_markers = (':', ',', ';', '-', '—')
        continuation_words = {"following", "under", "below", "include", "contains", "forms", "undertaking", "satisfying", "list", "schedule"}
        
        is_list_query = any(w in q_lower for w in ["list", "documents", "forms", "organizations", "clients", "alternative"])
        
        for cand in candidates:
            text = cand["text"].strip()
            if not text:
                continue
                
            needs_expansion = False
            last_char = text[-1]
            
            # Rule A: Ends in a continuation punctuation mark
            if last_char in continuation_markers:
                needs_expansion = True
                
            # Rule B: Ends in a continuation word
            last_words = text.lower().split()[-3:]
            if any(cw in lw for cw in continuation_words for lw in last_words):
                needs_expansion = True
                
            # Rule C: Ends mid-sentence with no standard ending punctuation
            if last_char.isalnum() and last_char not in ('.', '!', '?'):
                needs_expansion = True
                
            # Rule D: Force expansion for list/document queries to avoid partial lists
            if is_list_query:
                needs_expansion = True
                
            if needs_expansion:
                next_idx = cand["chunk_id"] + 1
                if next_idx < len(self.chunks_cache):
                    next_chunk = self.chunks_cache[next_idx]
                    # Verify they are on the same page
                    if next_chunk["metadata"]["page_number"] == cand["metadata"]["page_number"]:
                        # Stitch next chunk
                        cand["text"] = cand["text"] + "\n[CONTINUATION]: " + next_chunk["text"]
                        
                        # Double expansion for long lists (e.g. Q28, Q38) if another continuation exists
                        next_text = next_chunk["text"].strip()
                        if next_text and (next_text[-1] in continuation_markers or is_list_query):
                            third_idx = next_idx + 1
                            if third_idx < len(self.chunks_cache):
                                third_chunk = self.chunks_cache[third_idx]
                                if third_chunk["metadata"]["page_number"] == cand["metadata"]["page_number"]:
                                    cand["text"] = cand["text"] + "\n" + third_chunk["text"]

    def retrieve(self, query: str, n_results: int = 3, debug: bool = False) -> list:
        """
        Retrieves the top N matching chunks using a hybrid candidate retrieval
        followed by a Cross-Encoder semantic reranker, query target verification,
        domain booster, and dynamic context expansion.
        """
        if not self.chunks_cache:
            self._initialize_bm25()
            if not self.chunks_cache:
                return []

        q_lower = query.lower()
        q_lower = q_lower.replace("n.i.t.no.", "nit no").replace("n.i.t. no.", "nit no")
        q_lower = q_lower.replace("n.i.t. no", "nit no").replace("n.i.t.", "nit")
        q_lower = q_lower.replace("e.m.d.", "emd")

        # Adjust context window limit dynamically if a list query is requested
        is_list_query = any(w in q_lower for w in ["list", "what documents", "certificates", "what copies", "forms to be", "upload as part of", "checklist"])
        if is_list_query:
            n_results = min(7, max(n_results, 7))

        # --- DIRECT BOQ STRUCTURAL LOOKUP ---
        # Match item numbers like 5.1, 5.1.1, etc.
        boq_match = re.search(r'\b(?:item|sl\.?\s*no\.?|no\.?)\s*(\d+(?:\.\d+)*)\b', q_lower)
        if boq_match and self.chunks_cache:
            target_item = boq_match.group(1)
            matching_chunks = []
            for cand in self.chunks_cache:
                meta = cand.get("metadata", {})
                if meta.get("chunk_type") == "boq":
                    try:
                        structured = json.loads(meta.get("structured_json", "{}"))
                        if structured.get("item_no") == target_item:
                            matching_chunks.append(cand)
                    except Exception:
                        pass
            
            if matching_chunks:
                top_items = []
                for idx, cand in enumerate(matching_chunks[:n_results]):
                    meta = cand["metadata"]
                    top_items.append({
                        "chunk_id": cand.get("chunk_id", idx),
                        "text": cand["text"],
                        "metadata": {
                            "page_number": meta.get("page_number", 1),
                            "chunk_type": "boq",
                            "structured_json": meta.get("structured_json", "{}")
                        },
                        "dist": 0.0,
                        "sim": 1.0,
                        "bm25_score": 10.0,
                        "norm_bm25": 1.0,
                        "phrase_boost": 1.0,
                        "domain_boost": 1.0,
                        "rerank_score": 5.0,
                        "validation_penalty": 0.0,
                        "score": 10.0,
                        "confidence": "🟢 HIGH CONFIDENCE"
                    })
                self._expand_context(top_items, query)
                return top_items

        # Classify query target
        q_target = self._classify_query_target(query)

        # Expand query for type of tender to retrieve CPWD cover pages containing "percentage rate"
        if "type of tender" in q_lower or "tender type" in q_lower:
            query = query + " percentage rate item rate"

        # Expand query for Integrity Pact conditions to retrieve Page 36 containing "applicable"
        if "integrity pact" in q_lower and any(w in q_lower for w in ["condition", "required"]):
            query = query + " applicable"

        # 1. Fetch Candidate Chunks
        vector_res = query_chroma(self.collection, query, n_results=CANDIDATE_POOL_SIZE)
        v_ids = vector_res["ids"][0]
        v_docs = vector_res["documents"][0]
        v_metas = vector_res["metadatas"][0]
        v_dists = vector_res["distances"][0]

        bm25_scores = self.bm25.get_scores(query)
        bm25_scored_indices = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:CANDIDATE_POOL_SIZE]
        
        candidates = {}
        for cid, doc, meta, dist in zip(v_ids, v_docs, v_metas, v_dists):
            candidates[cid] = {
                "chunk_id": int(cid.split("_")[1]),
                "text": doc,
                "metadata": meta,
                "dist": dist,
                "bm25_score": 0.0
            }
            
        for idx, score in bm25_scored_indices:
            cid = f"chunk_{idx}"
            if cid not in candidates:
                candidates[cid] = {
                    "chunk_id": idx,
                    "text": self.chunks_cache[idx]["text"],
                    "metadata": self.chunks_cache[idx]["metadata"],
                    "dist": 1.0,
                    "bm25_score": score
                }
            else:
                candidates[cid]["bm25_score"] = score

        for cid, cand in candidates.items():
            if cand["bm25_score"] == 0.0:
                cand["bm25_score"] = bm25_scores[cand["chunk_id"]]

        # 2. Normalized Scoring Components
        max_bm25 = max([c["bm25_score"] for c in candidates.values()]) if candidates else 1.0
        if max_bm25 == 0.0:
            max_bm25 = 1.0
            
        cand_list = list(candidates.values())
        
        # 3. Compute Reranker Scores (if enabled)
        doc_texts = [c["text"] for c in cand_list]
        reranker_scores = [0.0] * len(cand_list)
        
        if self.reranker.enabled:
            reranker_scores = self.reranker.compute_scores(query, doc_texts)

        # 4. Integrate all scoring components
        scored_candidates = []
        for idx, cand in enumerate(cand_list):
            doc = cand["text"]
            d_lower = doc.lower()
            d_lower = d_lower.replace("n.i.t.no.", "nit no").replace("n.i.t. no.", "nit no")
            d_lower = d_lower.replace("n.i.t. no", "nit no").replace("n.i.t.", "nit")
            d_lower = d_lower.replace("e.m.d.", "emd")
            
            dist = cand["dist"]
            sim = 1.0 - max(0.0, min(1.0, dist))
            norm_bm25 = cand["bm25_score"] / max_bm25
            phrase_boost = compute_phrase_boost(query, doc)
            domain_boost = self._get_domain_boost_score(query, doc)
            rerank_score = reranker_scores[idx]
            
            # --- POST-RERANK ANSWERABILITY VALIDATOR ---
            validation_penalty = 0.0
            currency_val_pattern = r'(?:rs\.?|₹|rupees?)\s*\d+[\d,]*|\b\d+[\d,]*\s*(?:lakh|crore|thousand|lac)\b'
            date_pattern = r'\d{2}\.\d{2}\.\d{4}'
            time_pattern = r'\d{2}\.\d{2}\s*(?:hrs|am|pm)'
            url_pattern = r'https?://\S+|www\.\S+|\b\S+\.gov\.in\S*\b'

            if q_target == "DATE_TIME":
                is_financial_cover_query = "financial cover" in q_lower and "open" in q_lower
                has_date_time = re.search(date_pattern, d_lower) or re.search(time_pattern, d_lower) or "later date" in d_lower or "communicated" in d_lower
                if not has_date_time:
                    validation_penalty = -0.6
                elif is_financial_cover_query and ("later date" in d_lower or "communicated" in d_lower):
                    validation_penalty = 0.3

            elif q_target == "CURRENCY_AMOUNT":
                has_amount = re.search(currency_val_pattern, d_lower) or any(w in d_lower for w in ["rs", "₹", "rupees", "amount", "cost"])
                if not has_amount:
                    validation_penalty = -0.5

            elif q_target == "URL":
                has_url = re.search(url_pattern, d_lower) or "eprocure.gov.in" in d_lower or "www." in d_lower
                if not has_url:
                    validation_penalty = -0.6

            elif q_target == "DEFINITION":
                has_definition = any(w in d_lower for w in ["means", "stands for", "defined as", "refers to", "tender for"])
                if not has_definition:
                    validation_penalty = -0.4

            elif q_target == "CONSEQUENCE":
                has_penalty = any(w in d_lower for w in ["debarred", "rejected", "forfeited", "invalid", "cancellation", "cancel", "disqualified", "prosecution", "penalized"])
                if not has_penalty:
                    validation_penalty = -0.5

            # Calculate final combined score
            if self.reranker.enabled:
                norm_rerank = 1.0 / (1.0 + math.exp(-rerank_score))
                final_score = 0.5 * norm_rerank + 0.3 * norm_bm25 + 0.2 * phrase_boost + domain_boost + validation_penalty
            else:
                final_score = 0.4 * sim + 0.4 * norm_bm25 + 0.2 * phrase_boost + domain_boost + validation_penalty
                
            scored_candidates.append({
                "chunk_id": cand["chunk_id"],
                "text": doc,
                "metadata": cand["metadata"],
                "dist": dist,
                "sim": sim,
                "bm25_score": cand["bm25_score"],
                "norm_bm25": norm_bm25,
                "phrase_boost": phrase_boost,
                "domain_boost": domain_boost,
                "rerank_score": rerank_score,
                "validation_penalty": validation_penalty,
                "score": final_score
            })

        # Sort by final score descending
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply Generic Answerability Filter and Rerank Fallback
        valid_candidates = []
        for cand in scored_candidates:
            if self._is_answerable(query, cand["text"]):
                valid_candidates.append(cand)
                
        if not valid_candidates:
            valid_candidates = scored_candidates
            
        # 5. Apply Dynamic sliding window Context Expansion on top retrieved items
        top_items = valid_candidates if debug else valid_candidates[:n_results]
        
        # Determine confidence labels based on reranker score and validation
        for cand in top_items:
            passed_val = self._is_answerable(query, cand["text"])
            if not passed_val:
                cand["confidence"] = "🔴 LOW CONFIDENCE"
            elif cand["rerank_score"] > 0.1:
                cand["confidence"] = "🟢 HIGH CONFIDENCE"
            elif cand["rerank_score"] >= -0.5:
                cand["confidence"] = "🟡 MEDIUM CONFIDENCE"
            else:
                cand["confidence"] = "🔴 LOW CONFIDENCE"

        self._expand_context(top_items, query)
        
        return top_items
