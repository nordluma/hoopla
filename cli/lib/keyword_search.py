import math
import pickle
import os
import string

from collections import Counter, defaultdict
from nltk.stem import PorterStemmer

from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords

replament_table = str.maketrans("", "", string.punctuation)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)
        self.idx_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

    def build(self):
        for m in load_movies():
            doc_id = m["id"]
            self.docmap[doc_id] = m
            self.__add_document(doc_id, f"{m['title']} {m['description']}")

    def load(self):
        with open(self.idx_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(self.idx_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        if doc_id not in self.term_frequencies:
            return 0

        tokens = tokenize_text(term, load_stopwords())
        if len(tokens) != 1:
            raise ValueError("term must be a single token")

        return self.term_frequencies[doc_id][tokens[0]]

    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term, load_stopwords())
        if len(tokens) != 1:
            raise ValueError("term must be as single token")

        doc_count = len(self.docmap)
        term_doc_count = len(self.index[tokens[0]])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def __add_document(self, doc_id: int, text: str):
        stopwords = load_stopwords()
        tokens = tokenize_text(text, stopwords)
        for t in tokens:
            self.index[t].add(doc_id)

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        self.term_frequencies[doc_id].update(tokens)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()

    stopwords = load_stopwords()
    query_tokens = tokenize_text(query, stopwords)

    seen, matches = set(), []
    for token in query_tokens:
        res = idx.get_documents(token)
        for movie_id in res:
            if movie_id in seen:
                continue
            seen.add(movie_id)
            doc = idx.docmap[movie_id]
            if not doc:
                continue
            matches.append(doc)

        if len(matches) >= limit:
            break

    return matches


def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)


def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id, term)


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
