from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    matches = []
    for movie in movies:
        if preprocess_text(query) in preprocess_text(movie["title"]):
            matches.append(movie)
            if len(matches) >= limit:
                break

    return matches


def preprocess_text(text: str) -> str:
    return text.lower()
