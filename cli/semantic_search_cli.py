#!/usr/bin/env python3

import argparse

from lib.search_utils import DEFAULT_SEARCH_LIMIT
from lib.semantic_search import (
    embed_query_text,
    embed_text,
    search,
    verify_embeddings,
    verify_model,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")
    subparsers.add_parser(
        "verify_embeddings", help="Verify embeddings for the movie set"
    )

    embed_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    embed_parser.add_argument("text", type=str, help="Text to embed")

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Generate an embedding for a search query"
    )
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser(
        "search", help="Search movies with semantic similarity"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="Optional limit for search results",
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "embed_text":
            embed_text(args.text)
        case "search":
            search(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
