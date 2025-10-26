#!/usr/bin/env python3

import argparse

from lib.keyword_search import build_command, idf_command, search_command, tf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index")

    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency for given document ID and term"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser(
        "idf", help="Get the inverse document frequency for given term"
    )
    idf_parser.add_argument(
        "term", type=str, help="Term to get the inverse document frequency for"
    )

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built.")
        case "search":
            print(f"Searching for: {args.query}")
            matches = search_command(args.query)
            for i, movie in enumerate(matches, 1):
                print(f"{i}. ({movie['id']}) {movie['title']}")
        case "tf":
            tf = tf_command(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document '{args.doc_id}': {tf}")
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
