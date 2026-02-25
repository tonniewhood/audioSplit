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
import threading
from typing import Dict, List

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
_audio_lock = asyncio.Lock()
_chunk_lock = asyncio.Lock()
_channel_predictions: Dict[str, NDArray[np.float32]] = {}
_predicted_chunks: Dict[str, Dict[int, Dict[str, ci.PredictedChunk]]] = {}


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
    fused_predictions = np.zeros((len(ci.SoundClassifications), cc.MAX_CHUNK_SIZE), dtype=np.float32)
    return fused_predictions

async def fuse_and_forward(request_id: str, chunk_sources: Dict[str, ci.PredictedChunk]) -> None:
    """
    Fuse the predictions from different sources for a given chunk and append it to the channel predictions.
    If all chunks for the request are received, create an OutputAudioFile and send it to the Gateway.

    Args:
        request_id (str): The unique identifier for the current request.
        chunk_sources (Dict[str, ci.PredictedChunk]): The dictionary containing predicted chunks from different sources for the current chunk.
    """

    try:
        # Verify all chunks have consistent total_chunks value
        total_chunks_values = [chunk.total_chunks for chunk in chunk_sources.values()]
        assert len(set(total_chunks_values)) == 1, "All chunks must have the same total_chunks value"
        fused_chunk = fuse_predictions(chunk_sources)
        async with _audio_lock:
            request_predictions = _channel_predictions.setdefault(request_id, np.zeros_like(fused_chunk))
            np.concatenate((request_predictions, fused_chunk), axis=1)

            if len(request_predictions) == total_chunks_values[0]:
                output_audio = ci.OutputAudioFile(
                    request_id=request_id,
                    sample_rate=cc.SAMPLE_RATE,
                    num_channels=len(ci.SoundClassifications),
                    dtype=np.float32,
                    audio_data=request_predictions,
                )
            
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(cc.GATEWAY_URL, json=output_audio.model_dump())
                _channel_predictions.pop(request_id, None)
                
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
            threading.Thread(
                target=fuse_and_forward, args=(predicted_chunk.request_id, chunk_sources)
            ).start()

    return {"status": 200, "message": "ACK"}
