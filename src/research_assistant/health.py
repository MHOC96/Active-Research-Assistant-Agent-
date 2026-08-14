"""Runtime health checks for external services and configuration."""

from __future__ import annotations

from research_assistant.config import Settings, get_settings

_PLACEHOLDER_KEYS = {
    "GROQ_API_KEY": {"gsk_your_groq_api_key_here", ""},
    "GOOGLE_API_KEY": {"your_google_api_key_here", ""},
}


def validate_configuration(settings: Settings | None = None) -> list[str]:
    """Return human-readable configuration errors, if any."""
    settings = settings or get_settings()
    errors: list[str] = []

    if settings.groq_api_key in _PLACEHOLDER_KEYS["GROQ_API_KEY"]:
        errors.append("GROQ_API_KEY is missing or still set to the placeholder value.")
    if not settings.google_api_keys:
        errors.append(
            "GOOGLE_API_KEY / GOOGLE_API_KEYS is missing or still set to placeholder values."
        )

    return errors


def validate_external_services(settings: Settings | None = None) -> list[str]:
    """Probe Groq and Gemini APIs and return actionable errors."""
    settings = settings or get_settings()
    errors = validate_configuration(settings)
    if errors:
        return errors

    errors.extend(_probe_groq(settings))
    errors.extend(_probe_gemini(settings))
    return errors


def _probe_groq(settings: Settings) -> list[str]:
    try:
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)
        client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": "Reply with OK"}],
            max_tokens=4,
            temperature=0.0,
        )
    except Exception as exc:
        return [
            "Groq API check failed. Verify GROQ_API_KEY and model access "
            f"({settings.groq_model}): {exc}"
        ]
    return []


def _probe_gemini(settings: Settings) -> list[str]:
    from google.genai import types

    from research_assistant.embeddings.key_rotator import GoogleApiKeyRotator, is_rate_limit_error

    try:
        rotator = GoogleApiKeyRotator(settings.google_api_keys)
    except ValueError as exc:
        return [str(exc)]

    model = settings.gemini_embedding_model.removeprefix("models/")
    client = rotator.client()
    last_exc: Exception | None = None

    for _ in range(rotator.key_count):
        try:
            response = client.models.embed_content(
                model=model,
                contents="health check",
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=settings.embedding_dimension,
                ),
            )
            if not response.embeddings:
                return ["Gemini embedding check failed: empty response."]
            if len(response.embeddings[0].values) != settings.embedding_dimension:
                return [
                    "Gemini embedding check failed: unexpected dimension "
                    f"{len(response.embeddings[0].values)}."
                ]
            return []
        except Exception as exc:
            last_exc = exc
            if is_rate_limit_error(exc) and rotator.rotate():
                client = rotator.client()
                continue
            return [
                "Gemini API check failed. Verify GOOGLE_API_KEY / GOOGLE_API_KEYS from "
                f"Google AI Studio and that the Generative Language API is enabled: {exc}"
            ]

    return [
        "Gemini API check failed: all configured API keys are rate limited. "
        f"Last error: {last_exc}"
    ]
