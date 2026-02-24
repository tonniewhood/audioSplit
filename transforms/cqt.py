
"""
CQT Transform service for the audio processing pipeline.
This service is responsible for:
- Receiving `AudioChunk` inputs from the chunking stage.
- Computing Constant-Q Transform (CQT) features for each chunk.
- Forwarding `CQTChunk` outputs to downstream services.

The CQT service adheres to the following Contract:

Boundary: ** Chunking -> Transforms[CQT] **
Uses `AudioChunk` dataclass for incoming audio chunks

Boundary: ** Transforms[CQT] -> (Tone Identifier | Channel Predictor) **
Uses `CQTChunk` dataclass for outgoing CQT features

Inputs:
- `AudioChunk` dataclass for incoming audio chunks
Outputs:
- `CQTChunk` dataclass for outgoing CQT features
- Logging of processing steps and errors for observability

"""

import httpx
from fastapi import FastAPI

from common.logging_utils import setup_logging
from common.interfaces import AlertPayload

# Service objects
app = FastAPI()
logger = setup_logging("transforms-CQT")

CHANNEL_PREDICTOR_URL = "http://localhost:8005/api/channel_predictor"

@app.post("/api/cqt")
async def cqt(payload: AlertPayload):
    logger.info(f"Received alert: {payload.message}; Source: {payload.source}")
    trace = payload.trace + ["cqt:ACK"]
    cqt_payload = AlertPayload(
        request_id=payload.request_id,
        message=payload.message,
        trace=trace,
        source="cqt",
    )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(CHANNEL_PREDICTOR_URL, json=cqt_payload.model_dump())
    except Exception as exc:
        logger.exception("Forwarding failed")
        return {"message": f"FAIL: cqt forwarding error: {exc}"}

    return {"message": "ACK"}
