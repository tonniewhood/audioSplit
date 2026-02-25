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
import numpy as np
from fastapi import FastAPI

import common.constants as cc
import common.interfaces as ci
from common.logging_utils import setup_logging


# Service objects
app = FastAPI()
logger = setup_logging("transforms-CHROMA")


async def compute_chroma_features(chunk: ci.AudioChunk) -> ci.ChromaChunk:
    """
    Compute chroma features for the given audio chunk.

    Args:
        chunk (AudioChunk): The audio chunk to process.

    Returns:
        ChromaChunk: The computed chroma features.
    """
    # Placeholder for chroma computation logic
    chroma_features = np.zeros(cc.NUM_PITCH_CLASSES, dtype=np.float32)
    return ci.ChromaChunk(
        request_id=chunk.request_id,
        chunk_index=chunk.chunk_index,
        total_chunks=chunk.total_chunks,
        num_pitches=len(chroma_features),
        dtype=np.float32,
        pitch_classes=chroma_features,
    )


@app.post("/api/chroma")
async def chroma(chunk: ci.AudioChunk) -> ci.JSONResponse:
    """
    Endpoint to receive audio chunks, compute chroma features, and forward results to Tone Identifier and Channel Predictor.

    Args:
        chunk (AudioChunk): The incoming audio chunk containing waveform data and metadata.

    Returns:
        JSONResponse: A response indicating the status of the operation.
    """
    logger.info(f"Received chunk: {chunk.request_id}")

    try:
        chroma_chunk = await compute_chroma_features(chunk)

        logger.info(f"Computed chroma features for chunk: {chunk.request_id}, forwarding to downstream services.")

        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(cc.CHANNEL_PREDICTOR_URL, json={**chroma_chunk.model_dump(), "source": "chroma"})

    except Exception as exc:
        logger.exception(f"Data Validation failed: {exc}")
        return {
            "status": 500,
            "message": f"FAIL: Chroma processing error: {exc}",
        }

    return {
        "status": 200,
        "message": "ACK",
    }
