
"""
Channel Predictor service for the audio processing pipeline.
This service is responsible for:
- Receiving feature representations from transform services.
- Receiving tone predictions from the Tone Identifier.
- Producing per-class predictions for each audio chunk.

The Channel Predictor service adheres to the following Contract:

Boundary: ** Transforms[FFT|CQT|Chroma] -> Channel Predictor **
Uses `FFTChunk`, `CQTChunk`, and `ChromaChunk` dataclasses for incoming features

Boundary: ** Tone Identifier -> Channel Predictor **
Uses `TonePrediction` dataclass for incoming tone predictions

Boundary: ** Channel Predictor -> Channel Fuser **
Uses `PredictedChunk` dataclass for outgoing predictions

Inputs:
- `FFTChunk`, `CQTChunk`, and `ChromaChunk` dataclasses for incoming features
- `TonePrediction` dataclass for incoming tone predictions
Outputs:
- `PredictedChunk` dataclasses for outgoing predictions (should be 3 per chunk, one for each transform)
- Logging of processing steps and errors for observability

"""

from typing import Dict

import asyncio
import httpx
from fastapi import FastAPI

import common.constants as cc
import common.interfaces as ci
from common.logging_utils import setup_logging
from prediction.fft_predict import fft_predict_channels
from prediction.cqt_predict import cqt_predict_channels
from prediction.chroma_predict import chroma_predict_channels

EXPECTED_SOURCES = {"fft", "cqt", "chroma", "tone_identifier"}

# Service objects
app = FastAPI()
logger = setup_logging("prediction-CHANNEL_PREDICTOR")
_lock = asyncio.Lock()
_tone_predictions: Dict[str, ci.TonePrediction] = {}
_tone_events: Dict[str, asyncio.Event] = {}

async def get_tone_prediction(request_id: str, timeout: float = 10.0) -> ci.TonePrediction:
    async with _lock:
        event = _tone_events.get(request_id)
        if event is None:
            event = asyncio.Event()
            _tone_events[request_id] = event

    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"Timed out waiting for tone prediction: {request_id}") from exc

    async with _lock:
        prediction = _tone_predictions.get(request_id)
        if prediction is None:
            raise KeyError(f"Tone prediction missing for request_id: {request_id}")
        return prediction

@app.post("/api/fft/channel_predictor")
async def fft_channel_predictor(fft_chunk: ci.FFTChunk) -> ci.JSONResponse:
    """
    Recieve, validate, and store incoming FFT features
    
    Args:
        fft_chunk (ci.FFTChunk): The incoming FFT features for a specific audio chunk.
    
    Returns:
        JSONResponse: Acknowledgment of receipt and processing status.
    """
    try:
        fft_chunk.validate_contents()
    except AssertionError as exc:
        logger.error(f"FFT data validation failed for chunk: {fft_chunk.chunk_index} of request_id: {fft_chunk.request_id}: {exc}")
        return {
            "status": 400,
            "message": f"FAIL: FFT data validation error: {exc}"
        }
        
    logger.info(f"Received FFT features for chunk: {fft_chunk.chunk_index} of request_id: {fft_chunk.request_id}")

    try:
        tone_prediction = await get_tone_prediction(fft_chunk.request_id)
        tone_prediction.validate_contents()
        prediction = await fft_predict_channels(fft_chunk, tone_prediction)

        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(cc.CHANNEL_FUSER_URL, json=prediction.model_dump(mode="json"))

    except AssertionError as exc:
        logger.error(f"FFT channel prediction failed for chunk: {fft_chunk.chunk_index} of request_id: {fft_chunk.request_id}: {exc}")
        return {
            "status": 400,
            "message": f"FAIL: FFT channel prediction error: {exc}"
        }
    except Exception as exc:
        logger.exception(f"FFT channel prediction failed for chunk: {fft_chunk.chunk_index} of request_id: {fft_chunk.request_id}: {exc}")
        return {
            "status": 500,
            "message": f"FAIL: FFT channel prediction error: {exc}"
        }
    
    return {
        "status": 200,
        "message": "ACK; Features recieved"
    }

@app.post("/api/cqt/channel_predictor")
async def cqt_channel_predictor(cqt_chunk: ci.CQTChunk) -> ci.JSONResponse:
    """
    Recieve, validate, and store incoming CQT features
    
    Args:
        cqt_chunk (ci.CQTChunk): The incoming CQT features for a specific audio chunk.
    
    Returns:
        JSONResponse: Acknowledgment of receipt and processing status.
    """
    try:
        cqt_chunk.validate_contents()
    except AssertionError as exc:
        logger.error(f"CQT data validation failed for chunk: {cqt_chunk.chunk_index} of request_id: {cqt_chunk.request_id}: {exc}")
        return {
            "status": 400,
            "message": f"FAIL: CQT data validation error: {exc}"
        }
        
    logger.info(f"Received CQT features for chunk: {cqt_chunk.chunk_index} of request_id: {cqt_chunk.request_id}")

    try:
        tone_prediction = await get_tone_prediction(cqt_chunk.request_id)
        tone_prediction.validate_contents()
        prediction = await cqt_predict_channels(cqt_chunk, tone_prediction)

        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(cc.CHANNEL_FUSER_URL, json=prediction.model_dump(mode="json"))

    except AssertionError as exc:
        logger.error(f"CQT channel prediction failed for chunk: {cqt_chunk.chunk_index} of request_id: {cqt_chunk.request_id}: {exc}")
        return {
            "status": 400,
            "message": f"FAIL: CQT channel prediction error: {exc}"
        }
    except Exception as exc:
        logger.exception(f"CQT channel prediction failed for chunk: {cqt_chunk.chunk_index} of request_id: {cqt_chunk.request_id}: {exc}")
        return {
            "status": 500,
            "message": f"FAIL: CQT channel prediction error: {exc}"
        }

    return {
        "status": 200,
        "message": "ACK; Features recieved"
    }
    
@app.post("/api/chroma/channel_predictor")
async def chroma_channel_predictor(chroma_chunk: ci.ChromaChunk) -> ci.JSONResponse:
    """
    Recieve, validate, and store incoming chroma features
    
    Args:
        chroma_chunk (ci.ChromaChunk): The incoming chroma features for a specific audio chunk.
    
    Returns:
        JSONResponse: Acknowledgment of receipt and processing status.
    """
    try:
        chroma_chunk.validate_contents()
    except AssertionError as exc:
        logger.error(f"Chroma data validation failed for chunk: {chroma_chunk.chunk_index} of request_id: {chroma_chunk.request_id}: {exc}")
        return {
            "status": 400,
            "message": f"FAIL: Chroma data validation error: {exc}"
        }
        
    logger.info(f"Received chroma features for chunk: {chroma_chunk.chunk_index} of request_id: {chroma_chunk.request_id}")

    try:
        tone_prediction = await get_tone_prediction(chroma_chunk.request_id)
        tone_prediction.validate_contents()
        prediction = await chroma_predict_channels(chroma_chunk, tone_prediction)

        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(cc.CHANNEL_FUSER_URL, json=prediction.model_dump(mode="json"))

    except AssertionError as exc:
        logger.error(f"Chroma channel prediction failed for chunk: {chroma_chunk.chunk_index} of request_id: {chroma_chunk.request_id}: {exc}")
        return {
            "status": 400,
            "message": f"FAIL: Chroma channel prediction error: {exc}"
        }
    except Exception as exc:
        logger.exception(f"Chroma channel prediction failed for chunk: {chroma_chunk.chunk_index} of request_id: {chroma_chunk.request_id}: {exc}")
        return {
            "status": 500,
            "message": f"FAIL: Chroma channel prediction error: {exc}"
        }

    return {
        "status": 200,
        "message": "ACK; Features recieved"
    }

@app.post("/api/tone_identifier/channel_predictor")
async def tone_prediction_channel_predictor(tone_prediction: ci.TonePrediction) -> ci.JSONResponse:
    """
    Recieve, validate, and store incoming tone predictions
    
    Args:
        tone_prediction (ci.TonePrediction): The incoming tone prediction for a specific audio chunk.
    
    Returns:
        JSONResponse: Acknowledgment of receipt and processing status.
    """
    try:
        tone_prediction.validate_contents()
    except AssertionError as exc:
        logger.error(f"Tone Prediction data validation failed for request_id: {tone_prediction.request_id}: {exc}")
        return {
            "status": 400,
            "message": f"FAIL: Tone Prediction data validation error: {exc}"
        }
        
    logger.info(f"Received tone prediction for request_id: {tone_prediction.request_id}")

    async with _lock:
        _tone_predictions[tone_prediction.request_id] = tone_prediction
        event = _tone_events.get(tone_prediction.request_id)
        if event is None:
            event = asyncio.Event()
            _tone_events[tone_prediction.request_id] = event
        event.set()

    return {
        "status": 200,
        "message": "ACK; Tone Prediction recieved"
    }
    
