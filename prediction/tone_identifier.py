
"""
Tone Identifier service for the audio processing pipeline.
This service is responsible for:
- Receiving frequency-domain features from the transform services.
- Building a `Spectrogram` representation for tone analysis.
- Producing `TonePrediction` outputs for downstream services.

The Tone Identifier service adheres to the following Contract:

Boundary: ** Transforms[FFT] -> Tone Identifier **
Uses `FFTChunk` dataclass for incoming features

Boundary: ** (Internal) **
Uses `Spectrogram` dataclass for internal tone analysis

Boundary: ** Tone Identifier -> Channel Predictor **
Uses `TonePrediction` dataclass for outgoing predictions

Inputs:
- `FFTChunk` dataclass for incoming features
Outputs:
- `TonePrediction` dataclass for outgoing predictions
- Logging of processing steps and errors for observability
"""

import httpx
from fastapi import FastAPI

from common.logging_utils import setup_logging
from common.interfaces import AlertPayload

# Service objects
app = FastAPI()
logger = setup_logging("prediction-TONE_IDENTIFIER")

CHANNEL_PREDICTOR_URL = "http://localhost:8005/api/channel_predictor"

@app.post("/api/tone_identifier")
async def tone_identifier(payload: AlertPayload):
    logger.info(f"Received alert: {payload.message}; Source: {payload.source}")
    trace = payload.trace + ["tone_identifier:ACK"]
    tone_payload = AlertPayload(
        request_id=payload.request_id,
        message=payload.message,
        trace=trace,
        source="tone_identifier",
    )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(CHANNEL_PREDICTOR_URL, json=tone_payload.model_dump())
    except Exception as exc:
        logger.exception("Forwarding failed")
        return {"message": f"FAIL: tone_identifier forwarding error: {exc}"}

    return {"message": "ACK"}
