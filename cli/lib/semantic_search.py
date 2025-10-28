from sentence_transformers import SentenceTransformer
from torch import Tensor


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embedding(self, text: str) -> Tensor:
        if text.strip() == "":
            raise ValueError("text cannot be empty or only whitespace")

        return self.model.encode([text])[0]


def verify_model():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")
