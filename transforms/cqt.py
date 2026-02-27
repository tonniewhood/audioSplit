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

import asyncio
import httpx
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse

import common.constants as cc
import common.interfaces as ci
from common.logging_utils import setup_logging

# Service objects
app = FastAPI()
logger = setup_logging("transforms-CQT")
_cancelled: set[str] = set()


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
        valid_chunk=False,  # Placeholder while CQT computation is not implemented
        dtype="complex64",
        bins=cqt_features,
    )


async def _process_cqt_chunk(chunk: ci.AudioChunk) -> None:
    """
    Process a single audio chunk to compute CQT features and forward them to downstream services.

    Args:
        chunk (AudioChunk): The audio chunk to process.
    """
    if chunk.request_id in _cancelled:
        return
    try:
        # Build CQT features for the chunk
        cqt_chunk = await compute_cqt_features(chunk)

        # Forward features to the channel predictor
        logger.info(f"Computed CQT features for chunk: {chunk.request_id}, forwarding to downstream services.")
        async with httpx.AsyncClient(timeout=cc.HTTP_TIMEOUT) as client:
            await client.post(cc.build_predictor_url("cqt"), json=cqt_chunk.model_dump(mode="json"))

    except (httpx.ReadTimeout, httpx.ReadError):
        # Report downstream timeouts to the gateway
        logger.warning(f"CQT downstream timeout for chunk: {chunk.chunk_index} of request_id: {chunk.request_id}")
        error_payload = ci.ErrorPayload(
            request_id=chunk.request_id,
            source="cqt",
            message="CQT downstream timeout",
        )
        async with httpx.AsyncClient(timeout=cc.HTTP_ERROR_TIMEOUT) as client:
            await client.post(cc.GATEWAY_ERROR_URL, json=error_payload.model_dump(mode="json"))
    except Exception as exc:
        # Log unexpected failures for observability
        logger.exception(f"CQT processing failed for request_id: {chunk.request_id}: {exc}")


@app.post("/api/cqt")
async def cqt(chunk: ci.AudioChunk) -> JSONResponse:
    """
    Endpoint to receive audio chunks, compute CQT features, and forward results to Tone Identifier and Channel Predictor.

    Args:
        chunk (AudioChunk): The incoming audio chunk containing waveform data and metadata.

    Returns:
        JSONResponse: A response indicating the status of the operation.
    """
    if chunk.request_id in _cancelled:
        return JSONResponse(status_code=409, content={"status": 409, "message": "CANCELLED"})
    try:
        # Validate incoming payload
        chunk.validate_contents()

        # Acknowledge receipt and process asynchronously
        logger.info(f"Received chunk: {chunk.chunk_index} for request_id: {chunk.request_id}")
        asyncio.create_task(_process_cqt_chunk(chunk))
    except AssertionError as exc:
        # Validation errors are client errors
        logger.error(f"CQT data validation failed for chunk: {chunk.chunk_index} of request_id: {chunk.request_id}: {exc}")
        return JSONResponse(
            status_code=400,
            content={"status": 400, "message": f"FAIL: CQT data validation error: {exc}"},
        )
    except Exception as exc:
        # Unexpected errors are server errors
        logger.exception(f"CQT processing failed for chunk: {chunk.chunk_index} of request_id: {chunk.request_id}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"status": 500, "message": f"FAIL: CQT processing error: {exc}"},
        )

    return JSONResponse(status_code=200, content={"status": 200, "message": "ACK"})


@app.post("/api/cancel/{request_id}")
async def cancel(request_id: str) -> JSONResponse:
    """
    Cancel processing for the given request ID.

    Args:
        request_id (str): The unique identifier for the request to cancel.

    Returns:
        JSONResponse: A response indicating the status of the cancellation.
    """
    _cancelled.add(request_id)
    return JSONResponse(status_code=200, content={"status": 200, "message": "ACK"})
