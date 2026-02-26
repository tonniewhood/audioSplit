"""
Channel Fuser service for the audio processing pipeline.
This service is responsible for:
- Receiving predicted chunks from the Channel Predictor.
- Fusing per-class predictions into a final audio representation.
- Emitting `OutputAudioFile` for the Gateway to return to the web app.

The Channel Fuser service adheres to the following Contract:

Boundary: ** Channel Predictor -> Channel Fuser **
Uses `PredictedChunk` dataclass for incoming predictions

Boundary: ** Channel Fuser -> Gateway **
Uses `OutputAudioFile` dataclass for outgoing fused audio

Inputs:
- `PredictedChunk` dataclass for incoming predictions
Outputs:
- `OutputAudioFile` dataclass for outgoing fused audio
- Logging of processing steps and errors for observability
"""

import asyncio
from typing import Dict

import httpx
import numpy as np
from fastapi import FastAPI
from numpy.typing import NDArray

import common.constants as cc
import common.interfaces as ci
from common.logging_utils import setup_logging


# Service objects
app = FastAPI()
logger = setup_logging("prediction-CHANNEL_FUSER")
_chunk_lock = asyncio.Lock()
_fused_audio: Dict[str, ci.FusedAudio] = {}
_predicted_chunks: Dict[str, Dict[int, Dict[str, ci.PredictedChunk]]] = {}

SMOOTH_WINDOW = 128


def fuse_predictions(chunk_sources: Dict[str, ci.PredictedChunk]) -> NDArray[np.float32]:
    """
    Fuse the predictions from different sources for a given chunk into a single PredictedChunk.

    Args:
        chunk_sources (Dict[str, ci.PredictedChunk]): The dictionary containing predicted chunks from different sources for the current chunk.

    Returns:
        NDArray[np.float32]: The fused predicted chunk containing the combined predictions from all sources.
    """

    # Placeholder implementation for fusion logic
    # In a real implementation, this would involve combining the predictions in a meaningful way
    sample_len = next(iter(chunk_sources.values())).num_samples
    fused_predictions = np.zeros((len(ci.SoundClassifications), sample_len), dtype=np.float32)
    return fused_predictions


def smooth_predictions(predictions: NDArray[np.float32], window: int = SMOOTH_WINDOW) -> NDArray[np.float32]:
    """
    Apply a simple moving average smoothing to the predictions.
    
    Args:
        predictions (NDArray[np.float32]): The input predictions to smooth.
        window (int): The size of the moving average window.
        
    Returns:
        NDArray[np.float32]: The smoothed predictions.
    """
    if window <= 1:
        return predictions
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.zeros_like(predictions)
    for idx in range(predictions.shape[0]):
        smoothed[idx] = np.convolve(predictions[idx], kernel, mode="same")
    return smoothed

async def fuse_and_forward(request_id: str, chunk_sources: Dict[str, ci.PredictedChunk]) -> None:
    """
    Fuse the predictions from different sources for a given chunk and append it to the channel predictions.
    If all chunks for the request are received, create an OutputAudioFile and send it to the Gateway.

    Args:
        request_id (str): The unique identifier for the current request.
        chunk_sources (Dict[str, ci.PredictedChunk]): The dictionary containing predicted chunks from different sources for the current chunk.
    """

    try:
        # Verify all chunks have consistent total_chunks value and chunk_index
        chunk_indices = {chunk.chunk_index for chunk in chunk_sources.values()}
        total_chunks_values = {chunk.total_chunks for chunk in chunk_sources.values()}
        assert len(chunk_indices) == 1, "All sources must share the same chunk_index."
        assert len(total_chunks_values) == 1, "All chunks must have the same total_chunks value."

        chunk_index = next(iter(chunk_indices))
        total_chunks = next(iter(total_chunks_values))
        fused = fuse_predictions(chunk_sources)

        fused_chunk = ci.FusedChunk(
            request_id=request_id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            num_classes=len(ci.SoundClassifications),
            num_samples=fused.shape[1],
            dtype="float32",
            predictions=fused,
        )

        async with _chunk_lock:
            fused_audio = _fused_audio.get(request_id)
            if fused_audio is None:
                fused_audio = ci.FusedAudio(
                    request_id=request_id,
                    total_chunks=total_chunks,
                    chunks={},
                )
                _fused_audio[request_id] = fused_audio
            fused_audio.chunks[chunk_index] = fused_chunk

            if len(fused_audio.chunks) == fused_audio.total_chunks:
                ordered = [fused_audio.chunks[i].predictions for i in range(fused_audio.total_chunks)]
                full = np.concatenate(ordered, axis=1)
                full = smooth_predictions(full)

                output_audio = ci.OutputAudioFile(
                    request_id=request_id,
                    channels=len(ci.SoundClassifications),
                    num_samples=full.shape[1],
                    dtype="float32",
                    waveform=full,
                    sample_rate=cc.SAMPLE_RATE,
                )

                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(cc.GATEWAY_FINAL_URL, json=output_audio.model_dump(mode="json"))

                _fused_audio.pop(request_id, None)
            else:
                _fused_audio[request_id] = fused_audio
                
    except Exception as exc:
        logger.error(f"Error during fusion and forwarding for request_id {request_id}: {exc}")


@app.post("/api/channel_fuser")
async def channel_fuser(predicted_chunk: ci.PredictedChunk) -> ci.JSONResponse:
    """
    Recieve predicted chunk from Channel Predictor, store it, and if all predictions for the chunk are received, fuse them.
    The fused chunks are put together into an OutputAudioFile and sent to the Gateway.

    Args:
        predicted_chunk (PredictedChunk): The predicted chunk received from the Channel Predictor.

    Returns:
        JSONResponse: Acknowledgment of receipt and processing status.
    """
    try:
        predicted_chunk.validate_contents()
    except AssertionError as exc:
        logger.error(f"Invalid predicted chunk received: {exc}")
        return {"status": 400, "message": f"FAIL: Invalid predicted chunk: {exc}"}

    logger.info(
        f"Received predicted chunk: {predicted_chunk.chunk_index} from source: {predicted_chunk.prediction_source} for request_id: {predicted_chunk.request_id}"
    )
    async with _chunk_lock:
        request_chunks = _predicted_chunks.setdefault(predicted_chunk.request_id, {})
        chunk_sources = request_chunks.setdefault(predicted_chunk.chunk_index, {})
        chunk_sources[predicted_chunk.prediction_source] = predicted_chunk

        if len(chunk_sources) == cc.NUM_PREDICTORS:
            asyncio.create_task(fuse_and_forward(predicted_chunk.request_id, chunk_sources))
            request_chunks.pop(predicted_chunk.chunk_index, None)

    return {"status": 200, "message": "ACK"}
