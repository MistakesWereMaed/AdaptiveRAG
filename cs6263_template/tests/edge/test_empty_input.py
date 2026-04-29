"""Linguistic edge case tests."""
import pytest


def test_empty_input_does_not_crash():
    """The application must not crash on empty input."""
    from fastapi.testclient import TestClient
    from myproject.api import app
    with TestClient(app) as client:
        r = client.post("/api/query", json={"text": ""})
    assert r.status_code in (400, 422)


def test_very_long_input_does_not_crash():
    """The application must not crash on input far exceeding context window."""
    from fastapi.testclient import TestClient
    from myproject.api import app
    huge = "lorem ipsum " * 10_000
    with TestClient(app) as client:
        r = client.post("/api/query", json={"text": huge})
    # Either truncation succeeds (200) or the API rejects with 4xx.
    # Anything 5xx is a crash and fails this test.
    assert 200 <= r.status_code < 500


def test_non_ascii_input_does_not_crash():
    """The application must accept non-ASCII (e.g. CJK, Cyrillic, emoji) input."""
    from fastapi.testclient import TestClient
    from myproject.api import app
    with TestClient(app) as client:
        r = client.post("/api/query", json={"text": "你好世界 emoji 🌍"})
    assert 200 <= r.status_code < 500
