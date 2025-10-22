import string

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies

replament_table = str.maketrans("", "", string.punctuation)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    matches = []
    for movie in movies:
        query_tokens = tokenize_text(query)
        title_tokens = tokenize_text(movie["title"])

        if token_match(query_tokens, title_tokens) and movie not in matches:
            matches.append(movie)
            if len(matches) >= limit:
                break

    return matches


def preprocess_text(text: str) -> str:
    return text.lower().translate(replament_table)


def tokenize_text(text: str) -> list[str]:
    return list(filter(lambda t: t != "", preprocess_text(text).split()))


def token_match(query_tokens: list[str], tokens: list[str]) -> bool:
    return any(q in t for q in query_tokens for t in tokens)
