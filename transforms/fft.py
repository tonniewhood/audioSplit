"""
FFT Transform service for the audio processing pipeline.
This service is responsible for:
- Receiving `AudioChunk` inputs from the chunking stage.
- Computing FFT features for each chunk.
- Forwarding `FFTChunk` outputs to downstream services.

The FFT service adheres to the following Contract:

Boundary: ** Chunking -> Transforms[FFT] **
Uses `AudioChunk` dataclass for incoming audio chunks

Boundary: ** Transforms[FFT] -> (Tone Identifier | Channel Predictor) **
Uses `FFTChunk` dataclass for outgoing FFT features

Inputs:
- `AudioChunk` dataclass for incoming audio chunks
Outputs:
- `FFTChunk` dataclass for outgoing FFT features
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
logger = setup_logging("transforms-FFT")
_cancelled: set[str] = set()


async def compute_fft_features(chunk: ci.AudioChunk) -> ci.FFTChunk:
    """
    Compute FFT features for the given audio chunk.

    Args:
        chunk (AudioChunk): The audio chunk to process.

    Returns:
        FFTChunk: The computed FFT features.
    """
    # Placeholder for FFT computation logic

    try:
        # Compute FFT features for the chunk
        fft_features = np.fft.fft(chunk.waveform, n=cc.CHUNK_SIZE).astype(np.complex64)

        # Build the outgoing FFTChunk payload
        logger.info(
            f"Computed FFT features for chunk: {chunk.chunk_index} of request_id: {chunk.request_id}, forwarding to downstream services."
        )
        fft_chunk = ci.FFTChunk(
            request_id=chunk.request_id,
            chunk_index=chunk.chunk_index,
            total_chunks=chunk.total_chunks,
            num_bins=len(fft_features),
            bin_hz_resolution=chunk.sample_rate / cc.CHUNK_SIZE,
            valid_chunk=True,
            dtype="complex64",
            frequencies=fft_features,
            sample_rate=chunk.sample_rate,
        )

        # Forward FFT features downstream
        async with httpx.AsyncClient(timeout=cc.HTTP_TIMEOUT) as client:
            await asyncio.gather(
                client.post(cc.TONE_IDENTIFIER_URL, json=fft_chunk.model_dump(mode="json")),
                client.post(cc.build_predictor_url("fft"), json=fft_chunk.model_dump(mode="json")),
            )
    except (httpx.ReadTimeout, httpx.ReadError, asyncio.TimeoutError):
        # Report downstream timeouts to the gateway
        logger.warning(f"FFT downstream timeout for chunk: {chunk.chunk_index} of request_id: {chunk.request_id}")
        error_payload = ci.ErrorPayload(
            request_id=chunk.request_id,
            source="fft",
            message="FFT downstream timeout",
        )

        async with httpx.AsyncClient(timeout=cc.HTTP_ERROR_TIMEOUT) as client:
            await client.post(cc.GATEWAY_ERROR_URL, json=error_payload.model_dump(mode="json"))

        return JSONResponse(
            status_code=504,
            content={"status": 504, "message": "FAIL: FFT downstream timeout"},
        )

    except Exception as exc:
        # Surface unexpected failures for debugging
        logger.exception(f"Uncaught exception for request_id: {chunk.request_id}: {exc}")


@app.post("/api/fft")
async def fft(chunk: ci.AudioChunk) -> JSONResponse:
    """
    Endpoint to receive audio chunks, compute FFT features, and forward results to Tone Identifier and Channel Predictor.

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
        asyncio.create_task(compute_fft_features(chunk))

    except AssertionError as exc:
        # Validation errors are client errors
        logger.error(f"FFT data validation failed for chunk: {chunk.chunk_index} of request_id: {chunk.request_id}: {exc}")
        return JSONResponse(
            status_code=400,
            content={"status": 400, "message": f"FAIL: FFT data validation error: {exc}"},
        )

    except Exception as exc:
        # Unexpected errors are server errors
        logger.exception(f"FFT processing failed for chunk: {chunk.chunk_index} of request_id: {chunk.request_id}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"status": 500, "message": f"FAIL: FFT processing error: {exc}"},
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
