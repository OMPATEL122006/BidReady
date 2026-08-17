import re
from typing import List

class ExactSearchMatcher:
    """
    Computes generic exact lexical, phrase, and entity matching scores for 
    numbers, dates, percentages, names, monetary figures, and query-specific terms.
    """
    @staticmethod
    def compute_exact_score(query: str, doc_text: str) -> float:
        q_clean = re.sub(r'[^\w\s]', '', query.lower())
        d_clean = doc_text.lower()

        score = 0.0

        # 1. Full Query / Sub-phrase exact occurrence
        if len(q_clean) > 5 and q_clean in d_clean:
            score += 0.5

        # 2. Extract Stopword-filtered Key Tokens
        stopwords = {
            "what", "is", "the", "of", "for", "this", "do", "we", "need", "in", "a", "an", "to",
            "how", "much", "are", "on", "at", "by", "or", "and", "be", "with", "from", "which",
            "required", "applicable", "allowed", "details", "submitted", "when", "where", "who"
        }
        tokens = [w for w in q_clean.split() if w not in stopwords and len(w) > 2]

        if not tokens:
            return min(1.0, score)

        # 3. Dynamic 2-Word Bigram Intent Matching (e.g. "earnest money", "demand draft", "validity period")
        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
        matched_bigrams = [b for b in bigrams if b in d_clean]
        if bigrams:
            score += 0.3 * (len(matched_bigrams) / len(bigrams))

        # 4. Token Overlap Score
        matched_tokens = [t for t in tokens if t in d_clean]
        score += 0.2 * (len(matched_tokens) / len(tokens))

        # 5. Entity Presence Reward (numbers, percentages, amounts, date indicators in chunk)
        q_has_entities = bool(re.search(r'\b\d+(?:\.\d+)?%?\b|emd|nit|bid|cost|amount|date|time|period|day|month|year', q_clean))
        if q_has_entities:
            d_has_entity_val = bool(re.search(r'\b\d+(?:\.\d+)?%?\b|(?:rs\.?|₹)\s*\d+|\b(?:days|months|years|lakh|crore|nil|exempted|draft|cheque)\b', d_clean))
            if d_has_entity_val and matched_tokens:
                score += 0.2

        return min(1.0, score)
