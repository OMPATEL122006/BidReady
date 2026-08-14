import os
import requests
import json
from src.config import GROQ_API_KEY, DEFAULT_LLM_MODEL
from src.retriever import TenderRetriever

class TenderAnswerGenerator:
    """
    Connects the retrieval pipeline with the Groq LLM to generate
    accurate, structured, and page-cited answers based on the extracted
    tender document context.
    """
    def __init__(self, retriever: TenderRetriever = None):
        self.retriever = retriever or TenderRetriever()
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def generate_answer(self, query: str, n_results: int = 3) -> dict:
        """
        Retrieves relevant context elements and calls the Groq API to generate an answer.
        
        Returns:
            dict: {
                "answer": str,
                "sources": list of dict,
                "confidence": str
            }
        """
        if not GROQ_API_KEY:
            return {
                "answer": "Error: GROQ_API_KEY is not set. Please set it in your environment or .env file.",
                "sources": [],
                "confidence": "🔴 LOW CONFIDENCE"
            }

        # 1. Retrieve the top relevant chunks
        chunks = self.retriever.retrieve(query, n_results=n_results)
        if not chunks:
            return {
                "answer": "I couldn't find this information in the provided tender documents.",
                "sources": [],
                "confidence": "🔴 LOW CONFIDENCE"
            }

        # 2. Format context for LLM based on structural types
        context_parts = []
        sources = []
        max_confidence = "🔴 LOW CONFIDENCE"

        for idx, cand in enumerate(chunks):
            meta = cand.get("metadata", {})
            page = meta.get("page_number", "Unknown")
            c_type = meta.get("chunk_type", "text")
            text = cand["text"]
            conf = cand.get("confidence", "🟡 MEDIUM CONFIDENCE")
            
            # Determine maximum confidence
            if "🟢" in conf:
                max_confidence = "🟢 HIGH CONFIDENCE"
            elif "🟡" in conf and "🟢" not in max_confidence:
                max_confidence = "🟡 MEDIUM CONFIDENCE"

            # Render table layout for the LLM
            if c_type == "boq":
                try:
                    structured = json.loads(meta.get("structured_json", "{}"))
                    formatted_text = (
                        f"SOURCE: BOQ\n"
                        f"PAGE: {page}\n"
                        f"| Item | Description | Qty | Unit | Rate | Amount |\n"
                        f"|---|---|---|---|---|---|\n"
                        f"| {structured.get('item_no', '')} | {structured.get('description', '')} | "
                        f"{structured.get('quantity', '')} | {structured.get('unit', '')} | "
                        f"{structured.get('rate', '')} | {structured.get('amount', '')} |\n"
                    )
                except Exception:
                    formatted_text = f"SOURCE: BOQ\nPAGE: {page}\n{text}\n"
            elif c_type == "table":
                formatted_text = f"SOURCE: Tender Schedule (Table)\nPAGE: {page}\n{text}\n"
            else:
                formatted_text = f"SOURCE: Tender Clause\nPAGE: {page}\n{text}\n"
                
            context_parts.append(f"--- Evidence Unit {idx + 1} ---\n{formatted_text}")
            
            sources.append({
                "page": page,
                "text": text,
                "chunk_type": c_type,
                "confidence": conf
            })

        context_str = "\n\n".join(context_parts)

        # 3. Strict prompt matching all 15 rules & page citation requirements
        system_prompt = (
            "You are an expert procurement assistant specializing in Indian CPWD and government tenders.\n"
            "Your task is to answer the user's question accurately using ONLY the provided retrieved context chunks.\n\n"
            "Strict Guidelines:\n"
            "1. Answer ONLY from supplied evidence. Never use outside knowledge or make assumptions based on common government tender practices.\n"
            "2. If the evidence does not contain the answer, say: \"I couldn't find this information in the provided tender documents.\"\n"
            "3. If the tender says \"As per CPP Portal\" but the actual value/date/time is not present in the supplied evidence, do NOT invent or guess the value. "
            "Say: \"The tender refers to the CPP Portal for this information, but the actual value is not provided in the tender document.\"\n"
            "4. Distinguish between information not found in retrieved evidence, information explicitly stated as unavailable, and information that genuinely does not appear.\n"
            "5. If asked whether something is allowed:\n"
            "   - If the document explicitly says NO -> answer NO.\n"
            "   - If the document explicitly says YES -> answer YES.\n"
            "   - If there is no relevant clause -> say the tender does not specify it. Do NOT convert absence into 'No'.\n"
            "6. For numbers, preserve exact values. Never change ₹ amounts, percentages, quantities, dates, times, item numbers, or clause numbers.\n"
            "7. For BOQ questions, prefer structured BOQ evidence. When giving BOQ answers, preserve: item number, description, quantity, unit, rate, and amount.\n"
            "8. If calculation is requested, use only values explicitly provided in the evidence. Clearly distinguish \"Document value\" from \"Calculated value.\"\n"
            "9. If multiple tender clauses conflict, prefer the latest explicitly identified corrigendum/addendum, otherwise show the conflict rather than silently choosing.\n"
            "10. Trace every factual answer to the page numbers in the context using citations in format: [Page X] or [Page X, BOQ] or [Page X, Eligibility]. Be concise and professional."
        )

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"

        # 4. Invoke Groq API
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": DEFAULT_LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 1024
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                res_data = response.json()
                answer = res_data["choices"][0]["message"]["content"].strip()
                return {
                    "answer": answer,
                    "sources": sources,
                    "confidence": max_confidence
                }
            else:
                return {
                    "answer": f"Error: Groq API returned status code {response.status_code}. Response: {response.text}",
                    "sources": sources,
                    "confidence": "🔴 LOW CONFIDENCE"
                }
        except Exception as e:
            return {
                "answer": f"Error: Failed to connect to Groq API due to connection error: {str(e)}",
                "sources": sources,
                "confidence": "🔴 LOW CONFIDENCE"
            }
