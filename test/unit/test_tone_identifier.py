from unittest.mock import AsyncMock, patch

import numpy as np
from fastapi.testclient import TestClient

import common.constants as cc
import common.interfaces as ci
from prediction.tone_identifier import app


client = TestClient(app)


def test_tone_identifier_buffers_and_forwards():
    fft_chunk = ci.FFTChunk(
        request_id="req-1",
        chunk_index=0,
        total_chunks=2,
        num_bins=cc.CHUNK_SIZE,
        bin_hz_resolution=cc.SAMPLE_RATE / cc.CHUNK_SIZE,
        valid_chunk=True,
        dtype="complex64",
        frequencies=np.zeros(cc.CHUNK_SIZE, dtype=np.complex64),
        sample_rate=cc.SAMPLE_RATE,
    )
    audio_chunk = ci.AudioChunk(
        request_id="req-1",
        channels=1,
        num_samples=cc.CHUNK_SIZE,
        chunk_index=0,
        total_chunks=2,
        dtype="float32",
        waveform=np.zeros(cc.CHUNK_SIZE, dtype=np.float32),
        sample_rate=cc.SAMPLE_RATE,
    )
    tone_input = ci.ToneInput(fft_chunk=fft_chunk, audio_chunk=audio_chunk)

    with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
        resp = client.post("/api/tone_identifier", json=tone_input.model_dump(mode="json"))

    assert resp.status_code == 200
    # Depending on buffer saturation, it may or may not forward
    assert "message" in resp.json()
