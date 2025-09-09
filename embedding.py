from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[list]:
        return self.model.encode(texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True)
