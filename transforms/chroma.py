
"""
Chroma Transform service for the audio processing pipeline.
This service is responsible for:
- Receiving `AudioChunk` inputs from the chunking stage.
- Computing chroma features for each chunk.
- Forwarding `ChromaChunk` outputs to downstream services.

The Chroma service adheres to the following Contract:

Boundary: ** Chunking -> Transforms[Chroma] **
Uses `AudioChunk` dataclass for incoming audio chunks

Boundary: ** Transforms[Chroma] -> (Tone Identifier | Channel Predictor) **
Uses `ChromaChunk` dataclass for outgoing chroma features

Inputs:
- `AudioChunk` dataclass for incoming audio chunks
Outputs:
- `ChromaChunk` dataclass for outgoing chroma features
- Logging of processing steps and errors for observability
"""

import httpx
from fastapi import FastAPI

from common.logging_utils import setup_logging
from common.interfaces import AlertPayload

# Service objects
app = FastAPI()
logger = setup_logging("transforms-CHROMA")

CHANNEL_PREDICTOR_URL = "http://localhost:8005/api/channel_predictor"

@app.post("/api/chroma")
async def chroma(payload: AlertPayload):
    logger.info(f"Received alert: {payload.message}; Source: {payload.source}")
    trace = payload.trace + ["chroma:ACK"]
    chroma_payload = AlertPayload(
        request_id=payload.request_id,
        message=payload.message,
        trace=trace,
        source="chroma",
    )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(CHANNEL_PREDICTOR_URL, json=chroma_payload.model_dump())
    except Exception as exc:
        logger.exception("Forwarding failed")
        return {"message": f"FAIL: chroma forwarding error: {exc}"}

    return {"message": "ACK"}
