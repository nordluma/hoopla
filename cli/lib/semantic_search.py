import os
import re
import json
import numpy as np

from sentence_transformers import SentenceTransformer
from torch import Tensor

from .search_utils import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_SEMANTIC_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEMANTIC_CHUNK_OVERLAP,
    MOVIE_EMBEDDINGS_PATH,
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    format_search_result,
    load_movies,
)


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def search(self, query: str, limit: int):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first"
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first"
            )

        embedding = self.generate_embedding(query)

        res = [
            (cosine_similarity(embedding, doc_embedding), self.documents[i])
            for i, doc_embedding in enumerate(self.embeddings)
        ]
        res = sorted(res, key=lambda x: x[0], reverse=True)

        return list(
            map(
                lambda item: {
                    "score": item[0],
                    "title": item[1]["title"],
                    "description": item[1]["description"],
                },
                res[:limit],
            )
        )

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents

        doc_list = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(
            sentences=doc_list,
            show_progress_bar=True,
        )

        os.makedirs(MOVIE_EMBEDDINGS_PATH, exist_ok=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)

        return self.embeddings

    def generate_embedding(self, text: str) -> Tensor:
        text = text.strip()
        if text == "":
            raise ValueError("text cannot be empty or only whitespace")

        return self.model.encode([text])[0]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(
            CHUNK_METADATA_PATH
        ):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
                return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}

        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        chunk_metadata = []
        for idx, doc in enumerate(self.documents):
            text = doc.get("description", "")
            if not text.strip():
                continue

            chunks = semantic_chunking(text, max_chunk_size=4, overlap=1)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {
                        "movie_idx": idx,
                        "chunk_idx": i,
                        "total_chunks": len(chunks),
                    }
                )

        self.chunk_embeddings = self.model.encode(
            all_chunks,
            show_progress_bar=True,
        )
        self.chunk_metadata = chunk_metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)},
                f,
                indent=2,
            )

        return self.chunk_embeddings

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call `load_or_create_embeddings` first"
            )

        query_embedding = self.generate_embedding(query)
        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append(
                {
                    "chunk_idx": i,
                    "movie_idx": self.chunk_metadata[i]["movie_idx"],
                    "score": similarity,
                }
            )

        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if (
                movie_idx not in movie_scores
                or chunk_score["score"] > movie_scores[movie_idx]
            ):
                movie_scores[movie_idx] = chunk_score["score"]

        sorted_movies = sorted(
            movie_scores.items(), key=lambda item: item[1], reverse=True
        )

        res = []
        for movie_idx, score in sorted_movies[:limit]:
            doc = self.documents[movie_idx]
            res.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"][:100],
                    score=score,
                )
            )

        return res


def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0

    return dot_product / (norm1 * norm2)


def fixed_size_chunking(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    chunks = []
    words = text.split()

    i = 0
    n_words = len(words)
    while i < n_words - overlap:
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap

    return chunks


def semantic_chunking(
    text: str,
    max_chunk_size: int = DEFAULT_MAX_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_SEMANTIC_CHUNK_OVERLAP,
) -> list[str]:
    chunks = []
    parts = re.split(r"(?<=[.!?])\s+", text)

    i = 0
    n_sentences = len(parts)
    while i < n_sentences - overlap:
        chunk_sentences = parts[i : i + max_chunk_size]
        chunks.append(" ".join(chunk_sentences))
        i += max_chunk_size - overlap

    return chunks


def search_chunked_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    search = ChunkedSemanticSearch()
    search.load_or_create_chunk_embeddings(load_movies())
    results = search.search_chunks(query, limit)
    return {"query": query, "results": results}


def search(query: str, limit: int):
    search = SemanticSearch()
    search.load_or_create_embeddings(load_movies())
    for i, m in enumerate(search.search(query, limit), 1):
        print(f"{i}. {m['title']} (score: {m['score']:.4f})")
        print(f"\t{m['description'][:100]}...")
        print()


def semantic_chunk_text(
    text: str,
    max_chunk_size: int = DEFAULT_MAX_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_SEMANTIC_CHUNK_OVERLAP,
) -> None:
    chunks = semantic_chunking(text, max_chunk_size, overlap)

    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = fixed_size_chunking(text, chunk_size, overlap)

    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def embed_query_text(query: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def build_semantic_chunks() -> None:
    movies = load_movies()
    search = ChunkedSemanticSearch()
    print("Generating chunk embeddings")
    embeddings = search.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")


def verify_embeddings():
    search = SemanticSearch()
    documents = load_movies()
    embeddings = search.load_or_create_embeddings(documents)
    print(f"Number of docs: {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_text(text: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_model():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")
