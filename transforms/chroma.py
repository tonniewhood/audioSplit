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
logger = setup_logging("transforms-CHROMA")
_cancelled: set[str] = set()


async def compute_chroma_features(chunk: ci.AudioChunk) -> ci.ChromaChunk:
    """
    Compute chroma features for the given audio chunk.

    Args:
        chunk (AudioChunk): The audio chunk to process.

    Returns:
        ChromaChunk: The computed chroma features.
    """
    # Placeholder for chroma computation logic
    chroma_features = np.zeros(cc.CHUNK_SIZE, dtype=np.uint8)

    return ci.ChromaChunk(
        request_id=chunk.request_id,
        chunk_index=chunk.chunk_index,
        total_chunks=chunk.total_chunks,
        num_chroma=len(chroma_features),
        valid_chunk=False,  # Placeholder while chroma computation is not implemented
        dtype="uint8",
        pitch_classes=chroma_features,
    )


async def _process_chroma_chunk(chunk: ci.AudioChunk) -> None:
    """
    Process a single audio chunk to compute chroma features and forward them to downstream services.

    Args:
        chunk (AudioChunk): The audio chunk to process.
    """
    if chunk.request_id in _cancelled:
        return
    try:
        # Build chroma features for the chunk
        chroma_chunk = await compute_chroma_features(chunk)

        # Forward features to the channel predictor
        logger.info(f"Computed chroma features for chunk: {chunk.request_id}, forwarding to downstream services.")
        async with httpx.AsyncClient(timeout=cc.HTTP_TIMEOUT) as client:
            await client.post(cc.build_predictor_url("chroma"), json=chroma_chunk.model_dump(mode="json"))

    except (httpx.ReadTimeout, httpx.ReadError):
        # Report downstream timeouts to the gateway
        logger.warning(f"Chroma downstream timeout for chunk: {chunk.chunk_index} of request_id: {chunk.request_id}")
        error_payload = ci.ErrorPayload(
            request_id=chunk.request_id,
            source="chroma",
            message="Chroma downstream timeout",
        )
        async with httpx.AsyncClient(timeout=cc.HTTP_ERROR_TIMEOUT) as client:
            await client.post(cc.GATEWAY_ERROR_URL, json=error_payload.model_dump(mode="json"))
    except Exception as exc:
        # Log unexpected failures for observability
        logger.exception(f"Chroma processing failed for request_id: {chunk.request_id}: {exc}")


@app.post("/api/chroma")
async def chroma(chunk: ci.AudioChunk) -> JSONResponse:
    """
    Endpoint to receive audio chunks, compute chroma features, and forward results to Tone Identifier and Channel Predictor.

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
        asyncio.create_task(_process_chroma_chunk(chunk))
    except AssertionError as exc:
        # Validation errors are client errors
        logger.error(f"Chroma data validation failed for chunk: {chunk.chunk_index} of request_id: {chunk.request_id}: {exc}")
        return JSONResponse(
            status_code=400,
            content={"status": 400, "message": f"FAIL: Chroma data validation error: {exc}"},
        )
    except Exception as exc:
        # Unexpected errors are server errors
        logger.exception(f"Chroma processing failed for chunk: {chunk.chunk_index} of request_id: {chunk.request_id}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"status": 500, "message": f"FAIL: Chroma processing error: {exc}"},
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
