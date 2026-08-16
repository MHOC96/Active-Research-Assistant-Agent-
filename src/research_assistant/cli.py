"""CLI entry point."""

from __future__ import annotations

import argparse
import logging
import sys

from research_assistant.bootstrap import build_application
from research_assistant.citations.styles import list_citation_styles, parse_citation_style
from research_assistant.config import apply_fast_settings, get_settings
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
    parser.add_argument(
        "--citation-style",
        metavar="STYLE",
        help="Citation format: internal, apa7, mla9, chicago17, ieee, harvard",
    )
    parser.add_argument(
        "--list-citation-styles",
        action="store_true",
        help="List available citation styles and exit",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Speed mode: ingest 1 paper, larger chunks, batched embeddings (~1 min target)",
    )
    args = parser.parse_args()

    if args.list_citation_styles:
        print("Available citation styles:\n")
        for style_id, description in list_citation_styles():
            print(f"  {style_id:<12} {description}")
        return

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
        print(f"Citation style: {settings.citation_style}")
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
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("research_assistant.ingestion").setLevel(logging.INFO)
    logging.getLogger("research_assistant.pipeline").setLevel(logging.INFO)

    settings = apply_fast_settings(get_settings()) if args.fast else get_settings()

    if not args.skip_validation:
        errors = validate_external_services(settings)
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

    app = build_application(settings)
    citation_style = None
    if args.citation_style:
        try:
            citation_style = parse_citation_style(args.citation_style)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
    try:
        response = app.orchestrator.answer(
            args.query,
            citation_style=citation_style,
        )
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
