import re
from typing import List
from app.models.retrieval import QueryModel

class QueryExpander:
    """
    Generic query intent classifier and expander:
    - Categorizes queries into FACTUAL vs CONCEPTUAL.
    - Identifies target attributes and preferred document types.
    - Generates generic sub-queries and lexical variants.
    """
    def expand_query(self, query: str, document_id: str = None, tender_id: str = None) -> QueryModel:
        q_lower = query.lower()
        sub_queries = [query]

        target_type = "GENERAL"
        requested_attribute = "general"
        query_category = "FACTUAL"  # Default for tenders: most questions ask for specific factual terms
        preferred_doc_types = []

        # Categorize query intent: Conceptual vs Factual
        if any(k in q_lower for k in ["scope of work", "specification", "description of item", "general conditions", "procedure"]):
            query_category = "CONCEPTUAL"

        if any(k in q_lower for k in ["deadline", "last date", "submission date"]):
            target_type = "DATE_TIME"
            requested_attribute = "submission_deadline"
            preferred_doc_types = ["CORRIGENDUM", "NIT", "DETAILED_TENDER"]
        elif any(k in q_lower for k in ["emd amount", "earnest money deposit", "emd percentage", "emd"]):
            target_type = "CURRENCY_AMOUNT"
            requested_attribute = "emd_amount"
            preferred_doc_types = ["CORRIGENDUM", "NIT", "DETAILED_TENDER"]
        elif any(k in q_lower for k in ["mode", "form of emd", "demand draft", "payable", "payment mode"]):
            target_type = "GENERAL"
            requested_attribute = "emd_mode"
            preferred_doc_types = ["NIT", "DETAILED_TENDER"]
        elif any(k in q_lower for k in ["completion", "execution time", "time allowed"]):
            target_type = "DATE_TIME"
            requested_attribute = "completion_period"
            preferred_doc_types = ["NIT", "DETAILED_TENDER"]
        elif any(k in q_lower for k in ["bid validity", "validity period"]):
            target_type = "DATE_TIME"
            requested_attribute = "bid_validity"
            preferred_doc_types = ["NIT", "DETAILED_TENDER"]
        elif any(k in q_lower for k in ["performance security", "performance guarantee"]):
            target_type = "CURRENCY_AMOUNT"
            requested_attribute = "performance_security"
            preferred_doc_types = ["NIT", "DETAILED_TENDER"]
        elif any(k in q_lower for k in ["opening", "opened"]):
            target_type = "DATE_TIME"
            requested_attribute = "opening_date"
            preferred_doc_types = ["CORRIGENDUM", "NIT", "DETAILED_TENDER"]
        elif any(k in q_lower for k in ["joint venture", "jv", "consortium"]):
            target_type = "CONSEQUENCE"
            requested_attribute = "joint_venture_policy"
            preferred_doc_types = ["DETAILED_TENDER", "NIT"]
        elif any(k in q_lower for k in ["boq", "item", "quantity", "unit rate", "dsr"]):
            target_type = "BOQ"
            requested_attribute = "boq_item_quantity"
            preferred_doc_types = ["BOQ"]

        # Generic term expansions
        if "emd" in q_lower or "earnest money" in q_lower or "bid security" in q_lower:
            sub_queries.extend(["Earnest Money Deposit", "EMD amount", "Demand Draft EMD", "Bid Security"])

        if "completion" in q_lower or "period of completion" in q_lower:
            sub_queries.extend(["completion period", "execution period", "completed within", "time allowed"])

        if "tender number" in q_lower or "nit" in q_lower or "enquiry number" in q_lower:
            sub_queries.extend(["Tender Enquiry No", "Notice Inviting Tender", "NIT Ref"])

        return QueryModel(
            original_query=query,
            target_type=target_type,
            requested_attribute=requested_attribute,
            query_category=query_category,
            preferred_doc_types=preferred_doc_types,
            sub_queries=list(set(sub_queries)),
            tender_id=tender_id,
            document_id=document_id
        )
