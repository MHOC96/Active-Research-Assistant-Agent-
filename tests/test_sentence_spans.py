"""Tests for paste-to-cite sentence span offsets."""

from research_assistant.orchestrator.paste_to_cite import sentence_spans_in_text


def test_sentence_spans_preserve_leading_whitespace_offset():
    text = "  Cloud computing relies on containerization. Container engines enable scaling."
    spans = sentence_spans_in_text(text)

    assert len(spans) == 2
    assert spans[0][0] == "Cloud computing relies on containerization."
    assert text[spans[0][1] : spans[0][2]] == spans[0][0]
    assert text[spans[1][1] : spans[1][2]] == spans[1][0]


def test_sentence_spans_cover_full_paragraph():
    paragraph = (
        "Business owners are entrepreneurs until they become managers. "
        "When they become managers, they will feel frustrated because they will need to deal "
        "with new problems like managerial problems. "
        "And management is not a simple task. It needs knowledge and experience."
    )
    spans = sentence_spans_in_text(paragraph)

    assert len(spans) == 4
    for sentence, start, end in spans:
        assert paragraph[start:end] == sentence
