
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
from numpy.typing import NDArray

from common.logging_utils import setup_logging
from common.interfaces import AudioChunk, JSONResponse


# Constants
MAX_STR_LEN = 255
MAX_CHUNK_SIZE = 1024

# Downstream service URLs
TONE_IDENTIFIER_URL = "http://localhost:8004/api/tone_identifier"
CHANNEL_PREDICTOR_URL = "http://localhost:8005/api/channel_predictor"

# Service objects
app = FastAPI()
logger = setup_logging("transforms-FFT")


async def validate_audio_chunk(chunk: AudioChunk):
    """
    Validate the incoming AudioChunk against expected constraints.
    
    Args:
        chunk (AudioChunk): The audio chunk to validate.
        
    Raises:
        AssertionError: If any validation check fails.
    """
    assert chunk.request_id, "Request ID is required."
    assert len(chunk.request_id) <= MAX_STR_LEN, "Request ID is too long."
    assert chunk.channels == 1, "Only mono audio is supported."
    assert chunk.num_samples <= MAX_CHUNK_SIZE, f"Chunk size exceeds {MAX_CHUNK_SIZE} samples."
    assert chunk.total_chunks > 0, "Total chunks must be greater than zero."
    assert chunk.chunk_index >= 0, "Chunk index must be non-negative."
    assert chunk.chunk_index < chunk.total_chunks, "Chunk index must be less than total chunks."
    assert chunk.dtype == np.float32, "Audio samples must be float32."
    assert isinstance(chunk.waveform, NDArray[np.float32]), "Waveform must be a numpy array of float32."
    assert chunk.waveform.ndim == 1, "Waveform must be a 1D array."
    assert len(chunk.waveform) == chunk.num_samples, "Mismatch between num_samples and waveform length."

@app.post("/api/fft")
async def fft(chunk: AudioChunk) -> JSONResponse:
    """
    Endpoint to receive audio chunks, compute FFT features, and forward results to Tone Identifier and Channel Predictor.
    
    Args:
        chunk (AudioChunk): The incoming audio chunk containing waveform data and metadata.
        
    Returns:
        JSONResponse: A response indicating the status of the operation.
    """
    logger.info(f"Received chunk: {chunk.request_id}")

    try:
        
        validate_audio_chunk(chunk)
        fft_chunk = compute_fft_features(chunk)
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(TONE_IDENTIFIER_URL, json=fft_payload.model_dump())
            await client.post(CHANNEL_PREDICTOR_URL, json=fft_payload.model_dump())
    except Exception as exc:
        logger.exception("Forwarding failed")
        return {"message": f"FAIL: fft forwarding error: {exc}"}

    return {"message": "ACK"}
