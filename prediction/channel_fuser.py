

"""
Channel Fuser service for the audio processing pipeline.
This service is responsible for:
- Receiving predicted chunks from the Channel Predictor.
- Fusing per-class predictions into a final audio representation.
- Emitting `OutputAudioFile` for the Gateway to return to the web app.

The Channel Fuser service adheres to the following Contract:

Boundary: ** Channel Predictor -> Channel Fuser **
Uses `PredictedChunk` dataclass for incoming predictions

Boundary: ** Channel Fuser -> Gateway **
Uses `OutputAudioFile` dataclass for outgoing fused audio

Inputs:
- `PredictedChunk` dataclass for incoming predictions
Outputs:
- `OutputAudioFile` dataclass for outgoing fused audio
- Logging of processing steps and errors for observability
"""

import httpx
from fastapi import FastAPI

from common.logging_utils import setup_logging
from common.interfaces import AlertPayload

# Service objects
app = FastAPI()
logger = setup_logging("prediction-CHANNEL_FUSER")

GATEWAY_FINAL_URL = "http://localhost:8000/api/final"

@app.post("/api/channel_fuser")
async def channel_fuser(payload: AlertPayload):
    logger.info(f"Received alert: {payload.message}; Source: {payload.source}")
    trace = payload.trace + ["channel_fuser:ACK"]
    fuser_payload = AlertPayload(
        request_id=payload.request_id,
        message=payload.message,
        trace=trace,
        source="channel_fuser",
    )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(GATEWAY_FINAL_URL, json=fuser_payload.model_dump())
    except Exception as exc:
        logger.exception("Forwarding failed")
        return {"message": f"FAIL: channel_fuser forwarding error: {exc}"}

    return {"message": "ACK"}
