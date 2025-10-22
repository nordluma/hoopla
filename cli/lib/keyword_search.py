import string

from nltk.stem import PorterStemmer

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords

replament_table = str.maketrans("", "", string.punctuation)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    stopwords = load_stopwords()
    matches = []
    for movie in movies:
        query_tokens = tokenize_text(query, stopwords)
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
