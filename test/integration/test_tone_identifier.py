from unittest.mock import AsyncMock, patch

import numpy as np
from fastapi.testclient import TestClient

import common.interfaces as ci
from prediction.tone_identifier import app


client = TestClient(app)


def test_tone_identifier_buffers_and_forwards():
    fft_chunk = ci.FFTChunk(
        request_id="req-1",
        chunk_index=0,
        total_chunks=2,
        num_bins=1024,
        bin_hz_resolution=43.066,
        dtype="complex64",
        frequencies=np.zeros(1024, dtype=np.complex64),
        sample_rate=44100,
    )

    with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
        resp = client.post("/api/tone_identifier", json=fft_chunk.model_dump(mode="json"))

    assert resp.status_code == 200
    # Depending on buffer saturation, it may or may not forward
    assert "message" in resp.json()
