"""The grader verdict parser must be tolerant to small-model formatting."""

import pytest

from agentic_rag.graph.chains import parse_verdict


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The document mentions grading.\n\nVERDICT: yes", "yes"),
        ("Not related at all. VERDICT: no", "no"),
        ("verdict:yes", "yes"),
        ("Verdict: **no**", "no"),
        ("**VERDICT:** yes", "yes"),
    ],
)
def test_extracts_verdict(text, expected):
    assert parse_verdict(text, default="no") == expected


def test_falls_back_to_default_when_unparseable():
    assert parse_verdict("I am not sure what to say.", default="yes") == "yes"
    assert parse_verdict("", default="no") == "no"
