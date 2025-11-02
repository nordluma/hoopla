#!/usr/bin/env python3

import argparse

from lib.search_utils import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_SEMANTIC_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEMANTIC_CHUNK_OVERLAP,
)
from lib.semantic_search import (
    build_semantic_chunks,
    chunk_text,
    embed_query_text,
    embed_text,
    search,
    semantic_chunk_text,
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
    subparsers.add_parser(
        "embed_chunks",
        help="Generate semantically chunked embeddings for movie dataset",
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
        "search", help="Search for movies with semantic search"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of results to return",
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="Split text into fixed-size chunks with optional overlap"
    )
    chunk_parser.add_argument("query", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Size of each chunk in words",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Number of words to overlap between chunks",
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk",
        help="Split text into sentence chunks with optional overlap",
    )
    semantic_chunk_parser.add_argument("query", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=DEFAULT_MAX_SEMANTIC_CHUNK_SIZE,
        help="Number of sentenctes in each chunk",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_SEMANTIC_CHUNK_OVERLAP,
        help="Number of sentences to overlap between chunks",
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_chunks":
            build_semantic_chunks()
        case "embedquery":
            embed_query_text(args.query)
        case "embed_text":
            embed_text(args.text)
        case "search":
            search(args.query, args.limit)
        case "chunk":
            chunk_text(args.query, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_text(args.query, args.max_chunk_size, args.overlap)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
