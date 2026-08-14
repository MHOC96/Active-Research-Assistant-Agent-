"""CLI entry point."""

from __future__ import annotations

import argparse
import logging
import sys

from research_assistant.bootstrap import build_application
from research_assistant.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Active Research Assistant")
    parser.add_argument("--check-config", action="store_true", help="Validate configuration")
    parser.add_argument("query", nargs="?", help="Research question to answer")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable informational logging",
    )
    args = parser.parse_args()

    if args.check_config:
        settings = get_settings()
        print(f"Chroma collection: {settings.chroma_collection_name}")
        print(f"Embedding dimension: {settings.embedding_dimension}")
        print(f"Groq model: {settings.groq_model}")
        print(f"RRF weights: dense={settings.rrf_dense_weight}, sparse={settings.rrf_sparse_weight}")
        print(
            "Sufficiency: "
            f"MIN_CANDIDATES={settings.min_candidates}, "
            f"MIN_RERANK_SCORE={settings.min_rerank_score}"
        )
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    app = build_application()
    try:
        response = app.orchestrator.answer(args.query)
    finally:
        app.close()

    print(response.answer)
    if not response.citations_valid and response.citation_errors:
        print("\n[Citation validation warnings]", file=sys.stderr)
        for error in response.citation_errors:
            print(f"- {error}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
