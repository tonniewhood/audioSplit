import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from gateway.app import app

TEST_WAV = Path("test/resources/test-2048.wav")


@pytest.fixture()
def client():
    return TestClient(app)


def test_gateway_accepts_upload_and_fans_out(client):
    request_id = "req-2048"

    with open(TEST_WAV, "rb") as f, patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
        resp = client.post(
            "/api/alert",
            data={"request_id": request_id},
            files={"file": (TEST_WAV.name, f, "audio/wav")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "message" in body
    # Should fan out to FFT, CQT, Chroma
    assert mock_post.call_count == 6  # 3 transforms on a 2-chunk file


class TestGetPage:
    """
    Sanity check to make sure we're getting the expected HTML page from the root endpoint.

    Tests:
        - test_get_root_page: Verify that a GET request to the root endpoint returns a 200 status code and contains expected content.
    """

    def test_get_root_page(self, client: TestClient):
        """
        Verify that a GET request to the root endpoint returns a 200 status code and contains expected content.

        Args:
            client (TestClient): The test client fixture for making requests to the FastAPI app.
        """
        response = client.get("/")
        assert response.status_code == 200
        assert "AudioSplit Gateway" in response.text
