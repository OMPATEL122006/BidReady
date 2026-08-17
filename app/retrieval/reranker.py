from typing import List
from app.config.settings import USE_RERANKER, RERANKER_MODEL_NAME
from app.config.logging import logger

class CrossEncoderReranker:
    """
    Reranks top candidate chunks using SentenceTransformers Cross-Encoder model.
    """
    def __init__(self, model_name: str = RERANKER_MODEL_NAME, enabled: bool = USE_RERANKER):
        self.model_name = model_name
        self.enabled = enabled
        self.model = None

    def _init_model(self):
        if self.enabled and self.model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading Cross-Encoder Reranker model '{self.model_name}'...")
                self.model = CrossEncoder(self.model_name)
                logger.info("Reranker model loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load Cross-Encoder model: {e}. Falling back to hybrid scoring.")
                self.enabled = False

    def compute_scores(self, query: str, doc_texts: List[str]) -> List[float]:
        if not self.enabled or not doc_texts:
            return [0.0] * len(doc_texts)

        self._init_model()
        if not self.model:
            return [0.0] * len(doc_texts)

        try:
            pairs = [[query, doc] for doc in doc_texts]
            scores = self.model.predict(pairs)
            return [float(s) for s in scores]
        except Exception as e:
            logger.error(f"Error computing reranker scores: {e}")
            return [0.0] * len(doc_texts)
