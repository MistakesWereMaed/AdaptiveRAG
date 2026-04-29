"""Acceptance test for US-01: User submits a query and receives a cited answer.

The docstring of every user story test must contain the Given/When/Then from
docs/STORIES.md verbatim. The grading script reads test reports by story ID.
"""
import pytest


@pytest.mark.user_story("US-01")
def test_us_01_query_returns_cited_answer():
    """
    Given the application is running and the corpus is indexed,
    When the user submits the query "What is FIPS 140-3?",
    Then the response contains a non-empty answer string and at least one
    citation with a doc_id and snippet, and the latency is under 3 seconds.
    """
    # Replace this stub with your actual end-to-end test once the app exists.
    # For the regeneration test to be meaningful, this test must import from
    # src.myproject and exercise the same public interfaces declared in
    # docs/SPEC.md section 4.
    from myproject.router import route_query  # noqa: F401

    result = route_query("What is FIPS 140-3?")
    assert isinstance(result, dict)
    assert result.get("answer", "").strip() != ""
    assert len(result.get("citations", [])) >= 1
    for c in result["citations"]:
        assert "doc_id" in c
        assert "snippet" in c
    assert result.get("latency_ms", 99999) < 3000
