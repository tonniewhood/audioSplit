"""
Tone Identifier service for the audio processing pipeline.
This service is responsible for:
- Receiving frequency-domain features from the transform services.
- Building a `MelSpectrogram` representation for tone analysis.
- Producing `TonePrediction` outputs for downstream services.

The Tone Identifier service adheres to the following Contract:

Boundary: ** Transforms[FFT] -> Tone Identifier **
Uses `FFTChunk` dataclass for incoming features

Boundary: ** (Internal) **
Uses `MelSpectrogram` dataclass for internal tone analysis

Boundary: ** Tone Identifier -> Channel Predictor **
Uses `TonePrediction` dataclass for outgoing predictions

Inputs:
- `FFTChunk` dataclass for incoming features
Outputs:
- `TonePrediction` dataclass for outgoing predictions
- Logging of processing steps and errors for observability
"""

import asyncio
from typing import Dict

import httpx
import librosa
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from numpy.typing import NDArray
from pydantic import ValidationError

import common.constants as cc
import common.interfaces as ci
from common.logging_utils import setup_logging

# Service objects
app = FastAPI()
logger = setup_logging("prediction-TONE_IDENTIFIER")
chunk_buffers: Dict[str, ci.ChunkBuffer] = {}
_cancelled: set[str] = set()


async def build_mel_spectrogram(samples: NDArray[np.float32], sample_rate: int, request_id: str) -> ci.MelSpectrogram:
    """
    Build a mel spectrogram representation from a block of samples.

    Args:
        samples (NDArray[np.float32]): The buffer containing FFTChunks to process.
        sample_rate (int): The sample rate of the audio.
        request_id (str): The unique identifier for the current request.

    Returns:
        MelSpectrogram: The constructed mel spectrogram for tone analysis.
    """
    # Ensure we feed librosa a 1D float signal
    signal = np.asarray(samples)
    if signal.ndim > 1:
        signal = signal.reshape(-1)
    signal = np.abs(signal).astype(np.float32)

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_fft=cc.CHUNK_SIZE,
        hop_length=cc.MEL_HOP_LENGTH,
        n_mels=cc.MEL_N_MELS,
        fmin=0.0,
    ).astype(np.float32)

    return ci.MelSpectrogram(
        request_id=request_id,
        num_mels=cc.MEL_N_MELS,
        num_frames=mel.shape[1],
        hop_length=cc.MEL_HOP_LENGTH,
        dtype="float32",
        mel_spectrogram=mel,
        sample_rate=sample_rate,
    )


async def compute_tone_prediction(mel_spectrogram: ci.MelSpectrogram) -> ci.TonePrediction:
    """
    Compute a TonePrediction from the given mel spectrogram.

    Args:
        mel_spectrogram (MelSpectrogram): The mel spectrogram to analyze.

    Returns:
        TonePrediction: The predicted tone information for the audio chunk.
    """
    # Placeholder implementation
    predicted_tones = np.zeros((len(ci.SoundClassifications), 2), dtype=np.float32)  # pitch, confidence
    return ci.TonePrediction(
        request_id=mel_spectrogram.request_id,
        num_classes=len(ci.SoundClassifications),
        dtype="float32",
        class_probabilities=predicted_tones,
    )


@app.post("/api/tone_identifier")
async def tone_identifier(fft_features: ci.FFTChunk) -> JSONResponse:
    """
    Identifies the predominant tones/sound classifications within a given window of audio chunks.
    This endpoint receives FFT features, builds a mel spectrogram, computes tone predictions, and forwards results to the Channel Predictor.

    Args:
        fft_features (FFTChunk): The incoming FFT features containing frequency-domain information and metadata.

    Returns:
        JSONResponse: A response indicating the status of the operation.
    """
    if fft_features.request_id in _cancelled:
        return JSONResponse(status_code=409, content={"status": 409, "message": "CANCELLED"})

    # Initialize or fetch the rolling buffer for this request
    chunk_buffer = chunk_buffers.setdefault(fft_features.request_id, ci.ChunkBuffer(max_chunks=cc.MAX_CHUNK_BUFFER_SIZE))
    try:
        # Validate incoming payload
        fft_features.validate_contents()
    except (AssertionError, ValidationError) as exc:
        # Validation errors are client errors
        logger.exception(f"Data Validation failed: {exc}")
        return JSONResponse(
            status_code=400,
            content={"status": 400, "message": f"FAIL: Tone Identifier data validation error: {exc}"},
        )

    # Append the incoming FFT features to the buffer
    logger.info(
        f"Received FFT features for request_id: {fft_features.request_id}, "
        f"chunk_index: {fft_features.chunk_index}, "
        f"buffer_size: {chunk_buffer.num_chunks}/{chunk_buffer.max_chunks}"
    )
    chunk_buffer.append(fft_features.frequencies)

    # Decide whether to run the tone pipeline
    is_last_chunk = fft_features.chunk_index == (fft_features.total_chunks - 1)

    if not chunk_buffer.saturated and not is_last_chunk:
        logger.info(
            f"Tone buffer unsaturated for request_id: {fft_features.request_id} "
            f"({chunk_buffer.num_chunks}/{chunk_buffer.max_chunks})"
        )
        return JSONResponse(status_code=200, content={"status": 200, "message": "ACK; Buffer unsaturated"})

    if is_last_chunk:
        # Flush the remaining buffer on the final chunk
        logger.info(f"Tone buffer flush for request_id: {fft_features.request_id}; starting final pipeline")
        asyncio.create_task(_run_tone_pipeline(fft_features.request_id, chunk_buffer.get_block(), fft_features.sample_rate))
        chunk_buffer.clear()
        return JSONResponse(status_code=200, content={"status": 200, "message": "ACK; Tone pipeline started (flush)"})

    # Start a normal pipeline run when the buffer is saturated
    logger.info(f"Tone buffer saturated for request_id: {fft_features.request_id}; starting overlapping pipeline")
    asyncio.create_task(_run_tone_pipeline(fft_features.request_id, chunk_buffer.get_block(), fft_features.sample_rate))
    chunk_buffer.flush()

    return JSONResponse(status_code=200, content={"status": 200, "message": "ACK; Tone pipeline started"})


async def _run_tone_pipeline(request_id: str, samples: NDArray[np.float32], sample_rate: int) -> None:
    """
    Build a mel spectrogram and generate a tone prediction for a buffered window.

    Args:
        request_id (str): The unique identifier for the request.
        samples (NDArray[np.float32]): The buffered FFT feature window.
        sample_rate (int): The sample rate of the audio.
    """
    try:
        # Build the mel spectrogram for this window of audio
        logger.info(f"Building mel spectrogram for request_id: {request_id}")
        mel_spectrogram = await build_mel_spectrogram(samples, sample_rate, request_id)
        logger.info(f"Mel spectrogram ready for request_id: {request_id}")

        # Generate a tone prediction from the mel spectrogram
        tone_prediction = await compute_tone_prediction(mel_spectrogram)
        logger.info(f"Tone prediction computed for request_id: {request_id}")

        # Forward the prediction to the channel predictor
        async with httpx.AsyncClient(timeout=cc.HTTP_TIMEOUT) as client:
            await client.post(cc.build_predictor_url("tone_identifier"), json=tone_prediction.model_dump(mode="json"))
        logger.info(f"Tone prediction sent for request_id: {request_id}")
    except (httpx.ReadTimeout, httpx.ReadError):
        # Report downstream timeouts to the gateway
        logger.warning(f"Tone identifier downstream timeout for request_id: {request_id}")
        payload = ci.ErrorPayload(
            request_id=request_id,
            source="tone_identifier",
            message="Tone identifier downstream timeout",
        )
        try:
            async with httpx.AsyncClient(timeout=cc.HTTP_ERROR_TIMEOUT) as client:
                await client.post(cc.GATEWAY_ERROR_URL, json=payload.model_dump(mode="json"))
        except httpx.ReadTimeout:
            logger.warning(f"Tone identifier error reporting timeout for request_id: {request_id}")
        except Exception:
            logger.exception("Tone identifier failed to report timeout to gateway")
    except Exception:
        # Report unexpected failures to the gateway
        logger.exception(f"Tone identifier pipeline failed for request_id: {request_id}")
        payload = ci.ErrorPayload(
            request_id=request_id,
            source="tone_identifier",
            message="Tone identifier pipeline failed",
        )
        try:
            async with httpx.AsyncClient(timeout=cc.HTTP_ERROR_TIMEOUT) as client:
                await client.post(cc.GATEWAY_ERROR_URL, json=payload.model_dump(mode="json"))
        except httpx.ReadTimeout:
            logger.warning(f"Tone identifier error reporting timeout for request_id: {request_id}")
        except Exception:
            logger.exception("Tone identifier failed to report error to gateway")
    finally:
        pass


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
    chunk_buffers.pop(request_id, None)
    return JSONResponse(status_code=200, content={"status": 200, "message": "ACK"})
