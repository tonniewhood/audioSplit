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
import numpy as np
from fastapi import FastAPI

import common.constants as cc
import common.interfaces as ci
from common.logging_utils import setup_logging

# Service objects
app = FastAPI()
logger = setup_logging("transforms-CQT")


async def compute_cqt_features(chunk: ci.AudioChunk) -> ci.CQTChunk:
    """
    Compute CQT features for the given audio chunk.

    Args:
        chunk (AudioChunk): The audio chunk to process.

    Returns:
        CQTChunk: The computed CQT features.
    """
    # Placeholder for CQT computation logic
    cqt_features = np.zeros(cc.CHUNK_SIZE, dtype=np.complex64)
    return ci.CQTChunk(
        request_id=chunk.request_id,
        chunk_index=chunk.chunk_index,
        total_chunks=chunk.total_chunks,
        num_bins=len(cqt_features),
        bins_per_octave=12,
        f_min=32.703,
        dtype="complex64",
        bins=cqt_features,
    )


@app.post("/api/cqt")
async def cqt(chunk: ci.AudioChunk) -> ci.JSONResponse:
    """
    Endpoint to receive audio chunks, compute CQT features, and forward results to Tone Identifier and Channel Predictor.

    Args:
        chunk (AudioChunk): The incoming audio chunk containing waveform data and metadata.

    Returns:
        JSONResponse: A response indicating the status of the operation.
    """
    logger.info(f"Received chunk: {chunk.request_id}")

    try:
        cqt_chunk = await compute_cqt_features(chunk)

        logger.info(f"Computed CQT features for chunk: {chunk.request_id}, forwarding to downstream services.")

        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(cc.build_predictor_url("cqt"), json=cqt_chunk.model_dump(mode="json"))

    except Exception as exc:
        logger.exception(f"Data Validation failed: {exc}")
        return {
            "status": 500,
            "message": f"FAIL: CQT processing error: {exc}",
        }

    return {
        "status": 200,
        "message": "ACK",
    }
