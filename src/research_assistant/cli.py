"""Minimal CLI entry point."""

from __future__ import annotations

import argparse
import sys

from research_assistant.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Active Research Assistant")
    parser.add_argument("--check-config", action="store_true", help="Validate configuration")
    args = parser.parse_args()

    if args.check_config:
        settings = get_settings()
        print(f"Chroma collection: {settings.chroma_collection_name}")
        print(f"Embedding dimension: {settings.embedding_dimension}")
        print(f"RRF weights: dense={settings.rrf_dense_weight}, sparse={settings.rrf_sparse_weight}")
        print(f"Sufficiency: MIN_CANDIDATES={settings.min_candidates}, MIN_RERANK_SCORE={settings.min_rerank_score}")
        return

    print("Research assistant pipeline not yet wired. Use --check-config to verify setup.")
    sys.exit(0)


if __name__ == "__main__":
    main()
