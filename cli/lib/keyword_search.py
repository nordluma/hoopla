import pickle
import os
import string

from collections import defaultdict
from nltk.stem import PorterStemmer

from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords

replament_table = str.maketrans("", "", string.punctuation)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.idx_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    def build(self):
        for m in load_movies():
            doc_id = m["id"]
            self.docmap[doc_id] = m
            self.__add_document(doc_id, f"{m['title']} {m['description']}")

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(self.idx_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, text: str):
        stopwords = load_stopwords()
        tokens = tokenize_text(text, stopwords)
        for t in tokens:
            self.index[t].add(doc_id)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

    docs = idx.get_documents("merida")
    if len(docs) > 0:
        print(f"First document for token 'merida' = {docs[0]}")
    else:
        print("Not matches found")


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    stopwords = load_stopwords()
    query_tokens = tokenize_text(query, stopwords)

    matches = []
    for movie in movies:
        title_tokens = tokenize_text(movie["title"], stopwords)
        if token_match(query_tokens, title_tokens) and movie not in matches:
            matches.append(movie)
            if len(matches) >= limit:
                break

    return matches


def preprocess_text(text: str) -> str:
    return text.lower().translate(replament_table)


def tokenize_text(text: str, stopwords: list[str]) -> list[str]:
    tokens = list(filter(lambda t: t != "", preprocess_text(text).split()))
    filtered_tokens = remove_stopwords(tokens, stopwords)
    return stem_tokens(filtered_tokens)


def remove_stopwords(tokens: list[str], stopwords: list[str]) -> list[str]:
    stopwords_set = set(w for w in stopwords)
    return [t for t in tokens if t not in stopwords_set]


def stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return list(map(lambda t: stemmer.stem(t), tokens))


def token_match(query_tokens: list[str], tokens: list[str]) -> bool:
    return any(q in t for q in query_tokens for t in tokens)
