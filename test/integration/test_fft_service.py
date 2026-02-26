from unittest.mock import AsyncMock, patch

import numpy as np
from fastapi.testclient import TestClient

import common.interfaces as ci
from transforms.fft import app


client = TestClient(app)


def test_fft_forwards_to_tone_and_predictor():
    chunk = ci.AudioChunk(
        request_id="req-1",
        channels=1,
        num_samples=1024,
        chunk_index=0,
        total_chunks=2,
        dtype="float32",
        waveform=np.zeros(1024, dtype=np.float32),
        sample_rate=44100,
    )

    with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
        resp = client.post("/api/fft", json=chunk.model_dump(mode="json"))

    print(resp.status_code)
    print(resp.json())

    assert resp.status_code == 200
    assert resp.json()["message"] == "ACK"
    assert mock_post.call_count == 2
