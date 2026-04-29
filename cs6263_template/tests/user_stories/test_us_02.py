"""Acceptance test for US-02: Empty input shows an actionable error message."""
import pytest


@pytest.mark.user_story("US-02")
def test_us_02_empty_input_returns_error():
    """
    Given the application is running,
    When the user submits an empty query string,
    Then the API returns 400 with a message "input text is required",
    and no LLM call is made.
    """
    from fastapi.testclient import TestClient

    from myproject.api import app

    with TestClient(app) as client:
        response = client.post("/api/query", json={"text": ""})
    assert response.status_code == 400
    assert "input text is required" in response.json().get("error", "").lower()
