from unittest.mock import AsyncMock, patch

import numpy as np
from fastapi.testclient import TestClient

import common.interfaces as ci
from prediction.channel_predictor import app


client = TestClient(app)


def test_channel_predictor_waits_for_tone_prediction():
    tone = ci.TonePrediction(
        request_id="req-1",
        num_classes=len(ci.SoundClassifications),
        dtype="float32",
        class_probabilities=np.zeros((len(ci.SoundClassifications), 2), dtype=np.float32),
    )
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
        resp_tone = client.post("/api/tone_identifier/channel_predictor", json=tone.model_dump(mode="json"))
        assert resp_tone.status_code == 200
        resp_fft = client.post("/api/fft/channel_predictor", json=fft_chunk.model_dump(mode="json"))

    assert resp_fft.status_code == 200
    assert mock_post.call_count == 1
