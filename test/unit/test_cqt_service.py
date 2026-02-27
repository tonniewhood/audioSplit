from unittest.mock import AsyncMock, patch

import numpy as np
from fastapi.testclient import TestClient

import common.constants as cc
import common.interfaces as ci
from transforms.cqt import app


client = TestClient(app)


def test_cqt_forwards_to_predictor():
    chunk = ci.AudioChunk(
        request_id="req-1",
        channels=1,
        num_samples=cc.CHUNK_SIZE,
        chunk_index=0,
        total_chunks=2,
        dtype="float32",
        waveform=np.zeros(cc.CHUNK_SIZE, dtype=np.float32),
        sample_rate=44100,
    )

    with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
        resp = client.post("/api/cqt", json=chunk.model_dump(mode="json"))

    assert resp.status_code == 200
    assert resp.json()["message"] == "ACK"
    assert mock_post.call_count == 1
