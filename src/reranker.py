import logging
from sentence_transformers import CrossEncoder
from src.config import USE_RERANKER, RERANKER_MODEL_NAME

# Suppress verbose download warning messages if possible
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

class BGEReranker:
    def __init__(self):
        self.enabled = False
        self.model = None
        
        if USE_RERANKER:
            print(f"Loading Cross-Encoder Reranker model '{RERANKER_MODEL_NAME}'...")
            try:
                # Load the model locally using sentence-transformers CrossEncoder
                self.model = CrossEncoder(RERANKER_MODEL_NAME)
                self.enabled = True
                print("Reranker model loaded successfully.")
            except Exception as e:
                print(f"Warning: Failed to load Cross-Encoder Reranker model due to error: {e}")
                print("The system will automatically fallback to the local normalized hybrid keyword scoring.")
                self.enabled = False
        else:
            print("Cross-Encoder Reranker is disabled in config.py.")
            self.enabled = False

    def compute_scores(self, query: str, documents: list) -> list:
        """
        Computes relevance scores for a list of document strings against the query.
        
        Args:
            query (str): The search query.
            documents (list): List of document chunk texts.
            
        Returns:
            list: List of float relevance scores (higher is more relevant).
        """
        if not self.enabled or not self.model or not documents:
            return [0.0] * len(documents)
            
        # CrossEncoder expects a list of pairs: [[query, doc1], [query, doc2], ...]
        pairs = [[query, doc] for doc in documents]
        
        try:
            # Predict scores (typically values between 0.0 and 1.0 or logit scale, depending on the model)
            scores = self.model.predict(pairs)
            
            # Convert numpy float array to standard python floats list
            return [float(s) for s in scores]
        except Exception as e:
            print(f"Error during reranking inference: {e}")
            return [0.0] * len(documents)
