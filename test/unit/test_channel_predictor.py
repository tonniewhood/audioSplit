from unittest.mock import AsyncMock, patch

import numpy as np
from fastapi.testclient import TestClient

import common.constants as cc
import common.interfaces as ci
from prediction.channel_predictor import app


client = TestClient(app)


def test_channel_predictor_accepts_temporal_prediction():
    tone = ci.TonePrediction(
        request_id="req-1",
        num_classes=len(ci.SoundClassifications),
        dtype="float32",
        class_probabilities=np.zeros((len(ci.SoundClassifications), 2), dtype=np.float32),
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

    with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post, patch(
        "prediction.channel_predictor._process_temporal_prediction", new=AsyncMock()
    ):
        resp_tone = client.post("/api/tone_identifier/channel_predictor", json=tone.model_dump(mode="json"))
        assert resp_tone.status_code == 200
        resp_temporal = client.post("/api/temporal/channel_predictor", json=audio_chunk.model_dump(mode="json"))

    assert resp_temporal.status_code == 200
    assert mock_post.call_count == 0
