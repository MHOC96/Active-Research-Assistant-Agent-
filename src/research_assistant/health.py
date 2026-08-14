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
    if settings.google_api_key in _PLACEHOLDER_KEYS["GOOGLE_API_KEY"]:
        errors.append("GOOGLE_API_KEY is missing or still set to the placeholder value.")

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
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.google_api_key)
        model = settings.gemini_embedding_model.removeprefix("models/")
        response = client.models.embed_content(
            model=model,
            contents="health check",
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        if not response.embeddings:
            return ["Gemini embedding check failed: empty response."]
        if len(response.embeddings[0].values) != settings.embedding_dimension:
            return [
                "Gemini embedding check failed: unexpected dimension "
                f"{len(response.embeddings[0].values)}."
            ]
    except Exception as exc:
        return [
            "Gemini API check failed. Verify GOOGLE_API_KEY from Google AI Studio "
            f"and that the Generative Language API is enabled: {exc}"
        ]
    return []
