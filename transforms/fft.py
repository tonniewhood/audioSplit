
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

import common.constants as cc
import common.interfaces as ci
from common.logging_utils import setup_logging

# Service objects
app = FastAPI()
logger = setup_logging("transforms-FFT")


async def compute_fft_features(chunk: ci.AudioChunk) -> ci.FFTChunk:
    """
    Compute FFT features for the given audio chunk.
    
    Args:
        chunk (AudioChunk): The audio chunk to process.
        
    Returns:
        FFTChunk: The computed FFT features.
    """
    # Placeholder for FFT computation logic
    fft_features = np.fft.fft(chunk.waveform, n=cc.MAX_CHUNK_SIZE).astype(np.complex64)
    return ci.FFTChunk(
        request_id=chunk.request_id,
        chunk_index=chunk.chunk_index,
        total_chunks=chunk.total_chunks,
        num_bins=len(fft_features),
        bin_hz_resolution=chunk.sample_rate / cc.MAX_CHUNK_SIZE,
        dtype=np.complex64,
        frequencies=fft_features,
        sample_rate=chunk.sample_rate,
    )

@app.post("/api/fft")
async def fft(chunk: ci.AudioChunk) -> ci.JSONResponse:
    """
    Endpoint to receive audio chunks, compute FFT features, and forward results to Tone Identifier and Channel Predictor.
    
    Args:
        chunk (AudioChunk): The incoming audio chunk containing waveform data and metadata.
        
    Returns:
        JSONResponse: A response indicating the status of the operation.
    """
    try:
        chunk.validate_contents()
    except AssertionError as exc:
        logger.error(f"FFT data validation failed for chunk: {chunk.chunk_index} of request_id: {chunk.request_id}: {exc}")
        return {
            "status": 400,
            "message": f"FAIL: FFT data validation error: {exc}"
        }
    
    logger.info(f"Received chunk: {chunk.chunk_index} for request_id: {chunk.request_id}")

    try:
        fft_chunk = await compute_fft_features(chunk)
        
        logger.info(f"Computed FFT features for chunk: {chunk.chunk_index} of request_id: {chunk.request_id}, forwarding to downstream services.")
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            await asyncio.gather(
                client.post(cc.TONE_IDENTIFIER_URL, json=fft_chunk.model_dump()),
                client.post(cc.CHANNEL_PREDICTOR_URL, json={**fft_chunk.model_dump(), "source": "fft"})
            )
        
    except Exception as exc:
        logger.exception(f"FFT processing failed for chunk: {chunk.chunk_index} of request_id: {chunk.request_id}: {exc}")
        return {
            "status": 500,
            "message": f"FAIL: FFT processing error: {exc}"
        }

    return {
        "status": 200,
        "message": "ACK"
    }
