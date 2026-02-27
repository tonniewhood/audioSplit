from unittest.mock import AsyncMock, patch

import numpy as np
from fastapi.testclient import TestClient

import common.constants as cc
import common.interfaces as ci
from prediction.channel_fuser import app


client = TestClient(app)


def test_channel_fuser_accepts_predicted_chunks():
    pred = ci.PredictedChunk(
        request_id="req-1",
        chunk_index=0,
        total_chunks=2,
        num_classes=len(ci.SoundClassifications),
        num_samples=cc.CHUNK_SIZE,
        prediction_source="temporal",
        dtype="float32",
        predictions=np.zeros((len(ci.SoundClassifications), cc.CHUNK_SIZE), dtype=np.float32),
    )

    with patch("httpx.AsyncClient.post", new=AsyncMock()):
        resp = client.post("/api/channel_fuser", json=pred.model_dump(mode="json"))

    assert resp.status_code == 200
