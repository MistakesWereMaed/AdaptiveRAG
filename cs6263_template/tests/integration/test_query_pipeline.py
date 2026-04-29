"""Integration test stub.

Tests one end-to-end path: HTTP request through router, retriever, generator,
back to HTTP response. External LLM may be mocked but the wiring must be real.
"""


def test_query_pipeline_end_to_end():
    """Submitting a query exercises all pipeline stages."""
    from fastapi.testclient import TestClient

    from myproject.api import app

    with TestClient(app) as client:
        r = client.post("/api/query", json={"text": "test query"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "citations" in data
