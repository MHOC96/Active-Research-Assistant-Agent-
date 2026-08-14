"""CLI entry point."""

from __future__ import annotations

import argparse
import logging
import sys

from research_assistant.bootstrap import build_application
from research_assistant.config import get_settings
from research_assistant.health import validate_configuration, validate_external_services


def main() -> None:
    parser = argparse.ArgumentParser(description="Active Research Assistant")
    parser.add_argument("--check-config", action="store_true", help="Validate configuration")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="With --check-config, probe Groq and Gemini API connectivity",
    )
    parser.add_argument("--skip-validation", action="store_true", help="Skip API health checks")
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
        config_errors = validate_configuration(settings)
        if config_errors:
            print("\nConfiguration errors:")
            for error in config_errors:
                print(f"- {error}")
            sys.exit(1)
        if args.validate:
            print("\nProbing external services...")
            service_errors = validate_external_services(settings)
            if service_errors:
                print("\nService validation failed:")
                for error in service_errors:
                    print(f"- {error}")
                sys.exit(1)
            print("Groq and Gemini API checks passed.")
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.skip_validation:
        errors = validate_external_services()
        if errors:
            print("Pipeline startup validation failed:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            print(
                "\nFix the issues above, then rerun. "
                "Use --check-config --validate for details.",
                file=sys.stderr,
            )
            sys.exit(1)

    app = build_application()
    try:
        response = app.orchestrator.answer(args.query)
    except RuntimeError as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        sys.exit(1)
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
