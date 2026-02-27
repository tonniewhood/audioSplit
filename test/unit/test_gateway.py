import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import numpy as np
from fastapi.testclient import TestClient

from gateway.app import app

TEST_WAV = Path("test/resources/test-2048.wav")


@pytest.fixture()
def client() -> TestClient:
    """
    Provide a FastAPI test client for gateway integration tests.
    """
    return TestClient(app)


class TestGatewayAlert:
    """
    Test cases for the /api/alert endpoint.
    """

    def test_valid_upload_fans_out(self, client: TestClient) -> None:
        """
        Verify that a valid WAV upload is accepted and fanned out to all transforms.
        """
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
        # Should fan out to FFT, CQT, Temporal
        assert mock_post.call_count == 6  # 3 transforms on a 2-chunk file

    def test_invalid_sampling_frequency(self, client: TestClient) -> None:
        """
        Reject uploads when the decoded sampling rate is not 44.1 kHz.
        """
        request_id = "req-invalid-sr"

        with open(TEST_WAV, "rb") as f, patch("gateway.app.librosa_load", return_value=(np.zeros(2048, dtype=np.float32), 48000)):
            resp = client.post(
                "/api/alert",
                data={"request_id": request_id},
                files={"file": (TEST_WAV.name, f, "audio/wav")},
            )

        assert resp.status_code == 400

    def test_wrong_file_type(self, client: TestClient) -> None:
        """
        Reject uploads with disallowed file types.
        """
        request_id = "req-mkv"

        with open(TEST_WAV, "rb") as f:
            resp = client.post(
                "/api/alert",
                data={"request_id": request_id},
                files={"file": ("video.mkv", f, "video/x-matroska")},
            )

        assert resp.status_code == 400

    def test_malformed_wav(self, client: TestClient) -> None:
        """
        Reject uploads when decoding the WAV file fails.
        """
        request_id = "req-bad-wav"

        with open(TEST_WAV, "rb") as f, patch("gateway.app.librosa_load", side_effect=Exception("decode failed")):
            resp = client.post(
                "/api/alert",
                data={"request_id": request_id},
                files={"file": (TEST_WAV.name, f, "audio/wav")},
            )

        assert resp.status_code == 400

    def test_three_channel_input(self, client: TestClient) -> None:
        """
        Reject uploads that decode to multi-channel audio.
        """
        request_id = "req-3ch"
        fake_audio = np.zeros((3, 2048), dtype=np.float32)

        with open(TEST_WAV, "rb") as f, patch("gateway.app.librosa_load", return_value=(fake_audio, 44100)):
            resp = client.post(
                "/api/alert",
                data={"request_id": request_id},
                files={"file": (TEST_WAV.name, f, "audio/wav")},
            )

        assert resp.status_code == 400

    def test_22khz_sampling_frequency(self, client: TestClient) -> None:
        """
        Reject uploads when the decoded sampling rate is 22.05 kHz.
        """
        request_id = "req-22khz"

        with open(TEST_WAV, "rb") as f, patch("gateway.app.librosa_load", return_value=(np.zeros(2048, dtype=np.float32), 22050)):
            resp = client.post(
                "/api/alert",
                data={"request_id": request_id},
                files={"file": (TEST_WAV.name, f, "audio/wav")},
            )

        assert resp.status_code == 400

    def test_missing_request_id(self, client: TestClient) -> None:
        """
        Require the request_id form field.
        """
        with open(TEST_WAV, "rb") as f:
            resp = client.post(
                "/api/alert",
                files={"file": (TEST_WAV.name, f, "audio/wav")},
            )

        assert resp.status_code == 422

    def test_missing_file(self, client: TestClient) -> None:
        """
        Require the file form field.
        """
        resp = client.post(
            "/api/alert",
            data={"request_id": "req-no-file"},
        )

        assert resp.status_code == 422


class TestGetPage:
    """
    Sanity check to make sure we're getting the expected HTML page from the root endpoint.

    Tests:
        - test_get_root_page: Verify that a GET request to the root endpoint returns a 200 status code and contains expected content.
    """

    def test_get_root_page(self, client: TestClient) -> None:
        """
        Verify that a GET request to the root endpoint returns a 200 status code and contains expected content.

        Args:
            client (TestClient): The test client fixture for making requests to the FastAPI app.
        """
        response = client.get("/")
        assert response.status_code == 200
        assert "<!DOCTYPE html>" in response.text
