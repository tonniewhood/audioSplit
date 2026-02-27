
"""
Channel Predictor service for the audio processing pipeline.
This service is responsible for:
- Receiving feature representations from transform services.
- Receiving tone predictions from the Tone Identifier.
- Producing per-class predictions for each audio chunk.

The Channel Predictor service adheres to the following Contract:

Boundary: ** Transforms[FFT|CQT] / Temporal -> Channel Predictor **
Uses `FFTChunk`, `CQTChunk`, and `AudioChunk` dataclasses for incoming features

Boundary: ** Tone Identifier -> Channel Predictor **
Uses `TonePrediction` dataclass for incoming tone predictions

Boundary: ** Channel Predictor -> Channel Fuser **
Uses `PredictedChunk` dataclass for outgoing predictions

Inputs:
- `FFTChunk`, `CQTChunk`, and `AudioChunk` dataclasses for incoming features
- `TonePrediction` dataclass for incoming tone predictions
Outputs:
- `PredictedChunk` dataclasses for outgoing predictions (should be 3 per chunk, one for each transform)
- Logging of processing steps and errors for observability

"""

from typing import Dict

import asyncio
import httpx
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import ValidationError

import common.constants as cc
import common.interfaces as ci
from common.logging_utils import setup_logging
from prediction.fft_predict import fft_predict_channels
from prediction.cqt_predict import cqt_predict_channels
from prediction.temporal_predict import temporal_predict_channels

EXPECTED_SOURCES = {"fft", "cqt", "temporal", "tone_identifier"}

# Service objects
app = FastAPI()
logger = setup_logging("prediction-CHANNEL_PREDICTOR")
_lock = asyncio.Lock()
_tone_predictions: Dict[str, ci.TonePrediction] = {}
_cancelled: set[str] = set()

async def _report_error(request_id: str, source: str, message: str) -> None:
    """
    Report an error to the gateway for centralized handling.

    Args:
        request_id (str): The unique identifier for the request.
        source (str): The service or component reporting the error.
        message (str): A human-readable error message.
    """
    # Ship error details to the gateway for centralized handling
    payload = ci.ErrorPayload(request_id=request_id, source=source, message=message)
    try:
        async with httpx.AsyncClient(timeout=cc.HTTP_ERROR_TIMEOUT) as client:
            await client.post(cc.GATEWAY_ERROR_URL, json=payload.model_dump(mode="json"))
    except httpx.ReadTimeout:
        logger.warning(f"Error reporting timeout for request_id {request_id} (source={source})")
    except Exception:
        logger.exception(f"Failed to report error for request_id {request_id} (source={source})")

async def _get_or_init_tone_prediction(request_id: str) -> ci.TonePrediction:
    """
    Get the active tone prediction for a request, or initialize a uniform one.

    Args:
        request_id (str): The unique identifier for the request.

    Returns:
        TonePrediction: The active (or newly initialized) tone prediction.
    """
    async with _lock:
        prediction = _tone_predictions.get(request_id)
        if prediction is not None:
            return prediction

        # Initialize a uniform prediction until tone updates arrive
        num_classes = len(ci.SoundClassifications)
        uniform = np.full((num_classes, 2), 1.0 / num_classes, dtype=np.float32)
        prediction = ci.TonePrediction(
            request_id=request_id,
            num_classes=num_classes,
            dtype="float32",
            class_probabilities=uniform,
        )
        _tone_predictions[request_id] = prediction
        return prediction

async def _process_fft_prediction(fft_chunk: ci.FFTChunk) -> None:
    """
    Run FFT-based prediction for a chunk and forward results to the fuser.

    Args:
        fft_chunk (FFTChunk): The FFT features for the current chunk.
    """
    if fft_chunk.request_id in _cancelled:
        return
    try:
        # Resolve the active tone prediction for this request
        tone_prediction = await _get_or_init_tone_prediction(fft_chunk.request_id)
        tone_prediction.validate_contents()

        # Run the FFT-specific predictor
        prediction = await fft_predict_channels(fft_chunk, tone_prediction)

        # Forward results to the channel fuser
        async with httpx.AsyncClient(timeout=cc.HTTP_TIMEOUT) as client:
            await client.post(cc.CHANNEL_FUSER_URL, json=prediction.model_dump(mode="json"))

    except (AssertionError, ValidationError) as exc:
        # Validation errors indicate malformed payloads or outputs
        logger.error(f"FFT channel prediction failed for chunk: {fft_chunk.chunk_index} of request_id: {fft_chunk.request_id}: {exc}")
        await _report_error(fft_chunk.request_id, "channel_predictor", f"FFT validation error: {exc}")
    except (httpx.ReadTimeout, httpx.ReadError):
        # Downstream timeouts get reported to the gateway
        logger.warning(f"FFT downstream timeout for chunk: {fft_chunk.chunk_index} of request_id: {fft_chunk.request_id}")
        await _report_error(fft_chunk.request_id, "channel_predictor", "FFT downstream timeout")
    except Exception as exc:
        # Surface unexpected failures for debugging
        logger.exception(f"FFT channel prediction failed for chunk: {fft_chunk.chunk_index} of request_id: {fft_chunk.request_id}: {exc}")
        await _report_error(fft_chunk.request_id, "channel_predictor", f"FFT processing error: {exc}")

async def _process_cqt_prediction(cqt_chunk: ci.CQTChunk) -> None:
    """
    Run CQT-based prediction for a chunk and forward results to the fuser.

    Args:
        cqt_chunk (CQTChunk): The CQT features for the current chunk.
    """
    if cqt_chunk.request_id in _cancelled:
        return
    try:
        # Resolve the active tone prediction for this request
        tone_prediction = await _get_or_init_tone_prediction(cqt_chunk.request_id)
        tone_prediction.validate_contents()

        # Run the CQT-specific predictor
        prediction = await cqt_predict_channels(cqt_chunk, tone_prediction)

        # Forward results to the channel fuser
        async with httpx.AsyncClient(timeout=cc.HTTP_TIMEOUT) as client:
            await client.post(cc.CHANNEL_FUSER_URL, json=prediction.model_dump(mode="json"))

    except (AssertionError, ValidationError) as exc:
        # Validation errors indicate malformed payloads or outputs
        logger.error(f"CQT channel prediction failed for chunk: {cqt_chunk.chunk_index} of request_id: {cqt_chunk.request_id}: {exc}")
        await _report_error(cqt_chunk.request_id, "channel_predictor", f"CQT validation error: {exc}")
    except (httpx.ReadTimeout, httpx.ReadError):
        # Downstream timeouts get reported to the gateway
        logger.warning(f"CQT downstream timeout for chunk: {cqt_chunk.chunk_index} of request_id: {cqt_chunk.request_id}")
        await _report_error(cqt_chunk.request_id, "channel_predictor", "CQT downstream timeout")
    except Exception as exc:
        # Surface unexpected failures for debugging
        logger.exception(f"CQT channel prediction failed for chunk: {cqt_chunk.chunk_index} of request_id: {cqt_chunk.request_id}: {exc}")
        await _report_error(cqt_chunk.request_id, "channel_predictor", f"CQT processing error: {exc}")

async def _process_temporal_prediction(audio_chunk: ci.AudioChunk) -> None:
    """
    Run temporal prediction for a raw audio chunk and forward results to the fuser.

    Args:
        audio_chunk (AudioChunk): The raw audio chunk for the current window.
    """
    if audio_chunk.request_id in _cancelled:
        return
    try:
        # Resolve the active tone prediction for this request
        tone_prediction = await _get_or_init_tone_prediction(audio_chunk.request_id)
        tone_prediction.validate_contents()

        # Run the temporal predictor
        prediction = await temporal_predict_channels(audio_chunk, tone_prediction)

        # Forward results to the channel fuser
        async with httpx.AsyncClient(timeout=cc.HTTP_TIMEOUT) as client:
            await client.post(cc.CHANNEL_FUSER_URL, json=prediction.model_dump(mode="json"))

    except (AssertionError, ValidationError) as exc:
        # Validation errors indicate malformed payloads or outputs
        logger.error(
            f"Temporal channel prediction failed for chunk: {audio_chunk.chunk_index} "
            f"of request_id: {audio_chunk.request_id}: {exc}"
        )
        await _report_error(audio_chunk.request_id, "channel_predictor", f"Temporal validation error: {exc}")
    except (httpx.ReadTimeout, httpx.ReadError):
        # Downstream timeouts get reported to the gateway
        logger.warning(
            f"Temporal downstream timeout for chunk: {audio_chunk.chunk_index} of request_id: {audio_chunk.request_id}"
        )
        await _report_error(audio_chunk.request_id, "channel_predictor", "Temporal downstream timeout")
    except Exception as exc:
        # Surface unexpected failures for debugging
        logger.exception(
            f"Temporal channel prediction failed for chunk: {audio_chunk.chunk_index} "
            f"of request_id: {audio_chunk.request_id}: {exc}"
        )
        await _report_error(audio_chunk.request_id, "channel_predictor", f"Temporal processing error: {exc}")

@app.post("/api/fft/channel_predictor")
async def fft_channel_predictor(fft_chunk: ci.FFTChunk) -> JSONResponse:
    """
    Receive FFT features, validate the payload, and dispatch prediction work.
    
    Args:
        fft_chunk (ci.FFTChunk): The incoming FFT features for a specific audio chunk.
    
    Returns:
        JSONResponse: Acknowledgment of receipt and processing status.
    """
    if fft_chunk.request_id in _cancelled:
        return {"status": 409, "message": "CANCELLED"}
    if not getattr(fft_chunk, "chunk_valid", True):
        return JSONResponse(status_code=200, content={"status": 200, "message": "IGNORED: invalid chunk"})
    try:
        # Validate incoming payload
        fft_chunk.validate_contents()
    except (AssertionError, ValidationError) as exc:
        # Validation errors are client errors
        logger.error(f"FFT data validation failed for chunk: {fft_chunk.chunk_index} of request_id: {fft_chunk.request_id}: {exc}")
        return JSONResponse(
            status_code=400,
            content={"status": 400, "message": f"FAIL: FFT data validation error: {exc}"},
        )
        
    # Acknowledge receipt and process asynchronously
    logger.info(f"Received FFT features for chunk: {fft_chunk.chunk_index} of request_id: {fft_chunk.request_id}")

    asyncio.create_task(_process_fft_prediction(fft_chunk))
    return JSONResponse(status_code=200, content={"status": 200, "message": "ACK; Features recieved"})

@app.post("/api/cqt/channel_predictor")
async def cqt_channel_predictor(cqt_chunk: ci.CQTChunk) -> JSONResponse:
    """
    Receive CQT features, validate the payload, and dispatch prediction work.
    
    Args:
        cqt_chunk (ci.CQTChunk): The incoming CQT features for a specific audio chunk.
    
    Returns:
        JSONResponse: Acknowledgment of receipt and processing status.
    """
    if cqt_chunk.request_id in _cancelled:
        return {"status": 409, "message": "CANCELLED"}
    if not getattr(cqt_chunk, "chunk_valid", True):
        return JSONResponse(status_code=200, content={"status": 200, "message": "IGNORED: invalid chunk"})
    try:
        # Validate incoming payload
        cqt_chunk.validate_contents()
    except (AssertionError, ValidationError) as exc:
        # Validation errors are client errors
        logger.error(f"CQT data validation failed for chunk: {cqt_chunk.chunk_index} of request_id: {cqt_chunk.request_id}: {exc}")
        return JSONResponse(
            status_code=400,
            content={"status": 400, "message": f"FAIL: CQT data validation error: {exc}"},
        )
        
    # Acknowledge receipt and process asynchronously
    logger.info(f"Received CQT features for chunk: {cqt_chunk.chunk_index} of request_id: {cqt_chunk.request_id}")

    asyncio.create_task(_process_cqt_prediction(cqt_chunk))
    return JSONResponse(status_code=200, content={"status": 200, "message": "ACK; Features recieved"})
    
@app.post("/api/temporal/channel_predictor")
async def temporal_channel_predictor(audio_chunk: ci.AudioChunk) -> JSONResponse:
    """
    Receive raw audio chunks, validate the payload, and dispatch temporal prediction work.

    Args:
        audio_chunk (ci.AudioChunk): The incoming raw audio chunk.

    Returns:
        JSONResponse: Acknowledgment of receipt and processing status.
    """
    if audio_chunk.request_id in _cancelled:
        return {"status": 409, "message": "CANCELLED"}
    if not getattr(audio_chunk, "chunk_valid", True):
        return JSONResponse(status_code=200, content={"status": 200, "message": "IGNORED: invalid chunk"})
    try:
        # Validate incoming payload
        audio_chunk.validate_contents()
    except (AssertionError, ValidationError) as exc:
        # Validation errors are client errors
        logger.error(
            f"Temporal data validation failed for chunk: {audio_chunk.chunk_index} "
            f"of request_id: {audio_chunk.request_id}: {exc}"
        )
        return JSONResponse(
            status_code=400,
            content={"status": 400, "message": f"FAIL: Temporal data validation error: {exc}"},
        )

    # Acknowledge receipt and process asynchronously
    logger.info(f"Received temporal features for chunk: {audio_chunk.chunk_index} of request_id: {audio_chunk.request_id}")

    asyncio.create_task(_process_temporal_prediction(audio_chunk))
    return JSONResponse(status_code=200, content={"status": 200, "message": "ACK; Features recieved"})

@app.post("/api/tone_identifier/channel_predictor")
async def tone_prediction_channel_predictor(tone_prediction: ci.TonePrediction) -> JSONResponse:
    """
    Receive tone predictions, validate the payload, and update state.
    
    Args:
        tone_prediction (ci.TonePrediction): The incoming tone prediction for a specific audio chunk.
    
    Returns:
        JSONResponse: Acknowledgment of receipt and processing status.
    """
    if tone_prediction.request_id in _cancelled:
        return {"status": 409, "message": "CANCELLED"}
    try:
        # Validate incoming payload
        tone_prediction.validate_contents()
    except (AssertionError, ValidationError) as exc:
        # Validation errors are client errors
        logger.error(f"Tone Prediction data validation failed for request_id: {tone_prediction.request_id}: {exc}")
        return JSONResponse(
            status_code=400,
            content={"status": 400, "message": f"FAIL: Tone Prediction data validation error: {exc}"},
        )
        
    # Update the active tone prediction for this request
    logger.info(f"Received tone prediction for request_id: {tone_prediction.request_id}")

    async with _lock:
        _tone_predictions[tone_prediction.request_id] = tone_prediction

    return JSONResponse(status_code=200, content={"status": 200, "message": "ACK; Tone Prediction recieved"})


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
    _tone_predictions.pop(request_id, None)
    return JSONResponse(status_code=200, content={"status": 200, "message": "ACK"})
    
