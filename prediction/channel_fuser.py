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
from fastapi.responses import JSONResponse
from numpy.typing import NDArray
from pydantic import ValidationError

import common.constants as cc
import common.interfaces as ci
from common.logging_utils import setup_logging


# Service objects
app = FastAPI()
logger = setup_logging("prediction-CHANNEL_FUSER")
_chunk_lock = asyncio.Lock()
_fused_audio: Dict[str, ci.FusedAudio] = {}
_predicted_chunks: Dict[str, Dict[int, Dict[str, object]]] = {}
_cancelled: set[str] = set()

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
    sample_len = next(iter(chunk_sources.values())).num_samples
    fused_predictions = np.zeros((len(ci.SoundClassifications), sample_len), dtype=np.float32)
    return (
        chunk_sources["fft"].predictions if "fft" in chunk_sources else fused_predictions
    )  # For testing, use FFT predictions if available, otherwise return zeros


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

    # Apply a simple moving average per class
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

        # Fuse the predictions for this chunk
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
            # Store the fused chunk in the request buffer
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
                # Assemble all fused chunks into a full output file
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

                try:
                    # Send final audio to the gateway
                    async with httpx.AsyncClient(timeout=cc.HTTP_TIMEOUT) as client:
                        await client.post(cc.GATEWAY_FINAL_URL, json=output_audio.model_dump(mode="json"))
                except (httpx.ReadTimeout, httpx.ReadError):
                    logger.warning(f"Channel fuser downstream timeout for request_id {request_id}")
                    return

                _fused_audio.pop(request_id, None)
            else:
                _fused_audio[request_id] = fused_audio

    except Exception as exc:
        # Report fusion failures to the gateway
        logger.error(f"Error during fusion and forwarding for request_id {request_id}: {exc}")
        try:
            async with httpx.AsyncClient(timeout=cc.HTTP_TIMEOUT) as client:
                await client.post(
                    cc.GATEWAY_ERROR_URL,
                    json=ci.ErrorPayload(
                        request_id=request_id,
                        source="channel_fuser",
                        message=str(exc),
                    ).model_dump(mode="json"),
                )
        except (httpx.ReadTimeout, httpx.ReadError):
            logger.warning(f"Channel fuser error reporting timeout for request_id {request_id}")
        except Exception:
            logger.exception("Failed to report error to gateway")


@app.post("/api/channel_fuser")
async def channel_fuser(predicted_chunk: ci.PredictedChunk) -> JSONResponse:
    """
    Receive predicted chunks, store them, and fuse once all sources are present.

    Args:
        predicted_chunk (PredictedChunk): The predicted chunk received from the Channel Predictor.

    Returns:
        JSONResponse: Acknowledgment of receipt and processing status.
    """
    if predicted_chunk.request_id in _cancelled:
        return JSONResponse(status_code=409, content={"status": 409, "message": "CANCELLED"})
    try:
        # Track the chunk in the per-request cache
        logger.info(
            f"Received predicted chunk: {predicted_chunk.chunk_index} from source: {predicted_chunk.prediction_source} "
            f"for request_id: {predicted_chunk.request_id}"
        )
        async with _chunk_lock:
            request_chunks = _predicted_chunks.setdefault(predicted_chunk.request_id, {})
            entry = request_chunks.setdefault(
                predicted_chunk.chunk_index,
                {"seen": 0, "chunks": []},
            )
            entry["seen"] += 1

            # Only store valid chunks for fusion, but always count arrivals
            if getattr(predicted_chunk, "chunk_valid", True):
                predicted_chunk.validate_contents()
                entry["chunks"].append(predicted_chunk)

            # Dispatch fusion once all predictor sources arrive
            if entry["seen"] >= cc.NUM_PREDICTORS:
                chunk_list = entry["chunks"]
                if len(chunk_list) == cc.NUM_PREDICTORS:
                    chunk_sources = {chunk.prediction_source: chunk for chunk in chunk_list}
                    asyncio.create_task(fuse_and_forward(predicted_chunk.request_id, chunk_sources))
                else:
                    logger.info(
                        f"Skipping fusion for request_id: {predicted_chunk.request_id} "
                        f"chunk_index: {predicted_chunk.chunk_index}; "
                        f"valid_sources={len(chunk_list)}/{cc.NUM_PREDICTORS}"
                    )
                request_chunks.pop(predicted_chunk.chunk_index, None)
    except (AssertionError, ValidationError) as exc:
        # Validation errors are client errors
        logger.error(f"Invalid predicted chunk received: {exc}")
        return JSONResponse(
            status_code=400,
            content={"status": 400, "message": f"FAIL: Invalid predicted chunk: {exc}"},
        )
    except Exception as exc:
        # Unexpected errors are server errors
        logger.exception(f"Error validating predicted chunk: {exc}")
        return JSONResponse(
            status_code=500,
            content={"status": 500, "message": f"FAIL: Error validating predicted chunk: {exc}"},
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
    _fused_audio.pop(request_id, None)
    _predicted_chunks.pop(request_id, None)
    return JSONResponse(status_code=200, content={"status": 200, "message": "ACK"})
