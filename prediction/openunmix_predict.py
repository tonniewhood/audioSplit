"""
Houses the prediction pipeline for Open-Unmix (source separation) features.
"""

import numpy as np
import torch

import common.constants as cc
import common.interfaces as ci

_umx_model: torch.nn.Module | None = None
_umx_device: torch.device | None = None


def _get_umx() -> tuple[torch.nn.Module, torch.device]:
    """
    Load the Open-Unmix model once and reuse it for inference.

    Returns:
        Tuple containing the model and device.
    """
    global _umx_model, _umx_device
    if _umx_model is not None and _umx_device is not None:
        return _umx_model, _umx_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.hub.set_dir(cc.UNMIX_CACHE_DIR)
    model = torch.hub.load("sigsep/open-unmix-pytorch", "umxhq", device=device)
    _umx_model = model
    _umx_device = device
    return model, device


def _separate_sources(audio: np.ndarray) -> dict[str, np.ndarray]:
    """
    Run Open-Unmix separation and return a dict of stem arrays.

    Args:
        audio (np.ndarray): Mono audio array shaped (num_samples,).

    Returns:
        Dict mapping stem name to waveform (np.ndarray).
    """
    model, device = _get_umx()
    signal = np.asarray(audio, dtype=np.float32)
    if signal.ndim != 1:
        signal = signal.reshape(-1)

    # Open-Unmix expects (batch, channels, time). Use 2 channels for umxhq.
    stereo = np.stack([signal, signal], axis=0)
    tensor = torch.from_numpy(stereo).unsqueeze(0).to(device)

    if hasattr(model, "separate"):
        estimates = model.separate(tensor)
    else:
        estimates = model(tensor)

    sources: dict[str, np.ndarray] = {}
    if isinstance(estimates, dict):
        for name, wav in estimates.items():
            sources[name] = wav.squeeze().detach().cpu().numpy()
        return sources

    names = getattr(model, "sources", ["vocals", "drums", "bass", "other"])
    est = estimates
    # If a batch dimension is present, drop it
    if est.ndim >= 1 and est.shape[0] == 1:
        est = est.squeeze(0)

    # If still only one source returned, map it to "other"
    if est.ndim == 1 or (est.ndim >= 2 and est.shape[0] == 1):
        sources["other"] = est.squeeze().detach().cpu().numpy()
        return sources

    for idx, name in enumerate(names):
        if idx >= est.shape[0]:
            break
        sources[name] = est[idx].squeeze().detach().cpu().numpy()

    return sources


def _to_mono(wav: np.ndarray) -> np.ndarray:
    """
    Ensure a mono waveform for a source.
    """
    arr = np.asarray(wav, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return np.mean(arr, axis=0)
    return arr.reshape(-1)


async def openunmix_predict_channels(
    audio_chunk: ci.AudioChunk,
) -> ci.PredictedChunk:
    """
    Prediction logic using raw audio and tone predictions via Open-Unmix.

    Args:
        audio_chunk (ci.AudioChunk): The incoming raw audio chunk.
        tone_prediction (ci.TonePrediction): The tone prediction associated with the same request_id.

    Returns:
        PredictedChunk: The predicted channel information for the given chunk.
    """
    # Derive prediction dimensions from inputs
    num_classes = len(ci.SoundClassifications)
    num_samples = audio_chunk.num_samples

    # Run Open-Unmix separation and map stems to per-sample class channels
    sources = _separate_sources(audio_chunk.waveform)

    predictions = np.zeros((num_classes, num_samples), dtype=np.float32)

    stem_to_class = {
        "vocals": ci.SoundClassifications.VOCAL,
        "drums": ci.SoundClassifications.DRUMS,
        "bass": ci.SoundClassifications.BASS,
        "other": ci.SoundClassifications.OTHER,
    }

    for stem, cls in stem_to_class.items():
        if stem not in sources:
            continue
        wav = _to_mono(sources[stem])
        if len(wav) < num_samples:
            wav = np.pad(wav, (0, num_samples - len(wav)))
        elif len(wav) > num_samples:
            wav = wav[:num_samples]
        predictions[int(cls.value)] = wav.astype(np.float32)

    # Package the prediction for downstream consumption
    return ci.PredictedChunk(
        request_id=audio_chunk.request_id,
        chunk_index=audio_chunk.chunk_index,
        total_chunks=audio_chunk.total_chunks,
        num_classes=num_classes,
        num_samples=num_samples,
        prediction_source="temporal",
        dtype="float32",
        predictions=predictions,
        chunk_valid=getattr(audio_chunk, "chunk_valid", True),
    )


_umx_model, _umx_device = _get_umx()  # Preload the model at module import time for faster first inference
