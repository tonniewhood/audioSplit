"""
Tone Identifier service for the audio processing pipeline.
This service is responsible for:
- Receiving frequency-domain features from the transform services.
- Building a `MelSpectrogram` representation for tone analysis.
- Producing `TonePrediction` outputs for downstream services.

The Tone Identifier service adheres to the following Contract:

Boundary: ** Transforms[FFT] + Temporal -> Tone Identifier **
Uses `ToneInput` dataclass for incoming features

Boundary: ** (Internal) **
Uses `MelSpectrogram` dataclass for internal tone analysis

Boundary: ** Tone Identifier -> Channel Predictor **
Uses `TonePrediction` dataclass for outgoing predictions

Inputs:
- `ToneInput` dataclass for incoming features
Outputs:
- `TonePrediction` dataclass for outgoing predictions
- Logging of processing steps and errors for observability
"""

import asyncio
from typing import Dict

import httpx
import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, ASTForAudioClassification
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
_ast_model: ASTForAudioClassification | None = None
_ast_processor: AutoFeatureExtractor | None = None
_ast_device: torch.device | None = None


async def build_mel_spectrogram(samples: NDArray[np.float32], sample_rate: int, request_id: str) -> ci.MelSpectrogram:
    """
    Build a mel spectrogram representation from a block of audio samples.

    Args:
        samples (NDArray[np.float32]): The buffer containing audio samples to process.
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


def _get_ast() -> tuple[ASTForAudioClassification, AutoFeatureExtractor, torch.device]:
    """
    Load the AST model and processor once, using a local cache directory.

    Returns:
        Tuple containing the model, feature extractor, and device.
    """
    global _ast_model, _ast_processor, _ast_device
    if _ast_model is not None and _ast_processor is not None and _ast_device is not None:
        return _ast_model, _ast_processor, _ast_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoFeatureExtractor.from_pretrained(cc.AST_MODEL_ID, cache_dir=cc.AST_CACHE_DIR)
    model = ASTForAudioClassification.from_pretrained(cc.AST_MODEL_ID, cache_dir=cc.AST_CACHE_DIR)
    model.to(device)
    model.eval()

    _ast_model = model
    _ast_processor = processor
    _ast_device = device
    return model, processor, device


def _map_ast_to_classes(
    probs: torch.Tensor,
    id2label: dict[int, str],
) -> NDArray[np.float32]:
    """
    Map AST label probabilities to SoundClassifications.

    Args:
        probs (torch.Tensor): Softmax probabilities from AST with shape [num_labels].
        id2label (dict[int, str]): Mapping from label indices to names.

    Returns:
        NDArray[np.float32]: Array shaped (num_classes, 2) with [class_index, confidence].
    """
    num_classes = len(ci.SoundClassifications)
    out = np.zeros((num_classes, 2), dtype=np.float32)

    label_names = [id2label[i].lower() for i in range(len(id2label))]
    label_probs = probs.detach().cpu().numpy()

    def sum_for_keywords(keywords: list[str]) -> float:
        total = 0.0
        for idx, name in enumerate(label_names):
            if any(k in name for k in keywords):
                total += float(label_probs[idx])
        return total

    mapping = {
        ci.SoundClassifications.VOCAL: ["speech", "singing", "vocal"],
        ci.SoundClassifications.BASS: ["bass"],
        ci.SoundClassifications.DRUMS: ["drum", "drums", "percussion"],
        ci.SoundClassifications.OTHER: ["guitar", "piano", "keyboard", "synth", "other"],
    }

    for cls in ci.SoundClassifications:
        idx = int(cls.value)
        confidence = sum_for_keywords(mapping.get(cls, []))
        out[idx, 0] = float(idx)
        out[idx, 1] = float(confidence)

    return out


async def compute_tone_prediction(
    samples: NDArray[np.float32],
    mel_spectrogram: ci.MelSpectrogram,
) -> ci.TonePrediction:
    """
    Compute a TonePrediction from the given mel spectrogram using AST.

    Args:
        samples (NDArray[np.float32]): The raw audio samples for the window.
        mel_spectrogram (MelSpectrogram): The mel spectrogram to analyze.

    Returns:
        TonePrediction: The predicted tone information for the audio chunk.
    """
    # Run AST inference using the cached model
    model, processor, device = _get_ast()
    # AST expects raw waveform
    signal = np.asarray(samples)
    if signal.ndim > 1:
        signal = signal.reshape(-1)
    signal = signal.astype(np.float32)

    target_sr = getattr(processor, "sampling_rate", 16000)
    if mel_spectrogram.sample_rate != target_sr:
        signal = librosa.resample(signal, orig_sr=mel_spectrogram.sample_rate, target_sr=target_sr)

    features = processor(
        signal,
        sampling_rate=target_sr,
        return_tensors="pt",
        do_normalize=True,
    )
    features = {k: v.to(device) for k, v in features.items()}

    with torch.inference_mode():
        outputs = model(**features)
        probs = torch.softmax(outputs.logits.squeeze(0), dim=-1)

    predicted_tones = _map_ast_to_classes(probs, model.config.id2label)
    return ci.TonePrediction(
        request_id=mel_spectrogram.request_id,
        num_classes=len(ci.SoundClassifications),
        dtype="float32",
        class_probabilities=predicted_tones,
    )


@app.post("/api/tone_identifier")
async def tone_identifier(tone_input: ci.ToneInput) -> JSONResponse:
    """
    Identifies the predominant tones/sound classifications within a given window of audio chunks.
    This endpoint receives FFT features and raw audio, builds a mel spectrogram, computes tone predictions,
    and forwards results to the Channel Predictor.

    Args:
        tone_input (ToneInput): The incoming FFT features and raw audio chunk.

    Returns:
        JSONResponse: A response indicating the status of the operation.
    """
    fft_features = tone_input.fft_chunk
    audio_chunk = tone_input.audio_chunk

    if fft_features.request_id in _cancelled:
        return JSONResponse(status_code=409, content={"status": 409, "message": "CANCELLED"})

    # Initialize or fetch the rolling buffer for this request
    chunk_buffer = chunk_buffers.setdefault(fft_features.request_id, ci.ChunkBuffer(max_chunks=cc.MAX_CHUNK_BUFFER_SIZE))
    try:
        # Validate incoming payload
        tone_input.validate_tone_input()
    except (AssertionError, ValidationError) as exc:
        # Validation errors are client errors
        logger.exception(f"Data Validation failed: {exc}")
        return JSONResponse(
            status_code=400,
            content={"status": 400, "message": f"FAIL: Tone Identifier data validation error: {exc}"},
        )

    # Append the raw audio samples to the rolling buffer
    logger.info(
        f"Received tone input for request_id: {fft_features.request_id}, "
        f"chunk_index: {fft_features.chunk_index}, "
        f"buffer_size: {chunk_buffer.num_chunks}/{chunk_buffer.max_chunks}"
    )
    chunk_buffer.append(audio_chunk.waveform.astype(np.float32))

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
        asyncio.create_task(_run_tone_pipeline(fft_features.request_id, chunk_buffer.get_block(), audio_chunk.sample_rate))
        chunk_buffer.clear()
        return JSONResponse(status_code=200, content={"status": 200, "message": "ACK; Tone pipeline started (flush)"})

    # Start a normal pipeline run when the buffer is saturated
    logger.info(f"Tone buffer saturated for request_id: {fft_features.request_id}; starting overlapping pipeline")
    asyncio.create_task(_run_tone_pipeline(fft_features.request_id, chunk_buffer.get_block(), audio_chunk.sample_rate))
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
        tone_prediction = await compute_tone_prediction(samples, mel_spectrogram)
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


_ast_model, _ast_processor, _ast_device = _get_ast()  # Preload the model at module import time for faster first inference
