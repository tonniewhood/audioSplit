import asyncio
from pathlib import Path

import numpy as np
from librosa import load as librosa_load

import common.constants as cc
import common.interfaces as ci
from prediction.openunmix_predict import openunmix_predict_channels


TEST_FILE = Path("test/resources/test-blues-10s.wav")


async def run() -> None:
    """
    Load a test audio file, build an AudioChunk, and print Open-Unmix predictions.
    """
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Missing test file: {TEST_FILE}")

    data, sample_rate = librosa_load(TEST_FILE, sr=cc.SAMPLE_RATE, mono=True)
    num_samples = min(len(data), cc.CHUNK_SIZE)
    waveform = data[:num_samples].astype(np.float32)

    chunk = ci.AudioChunk(
        request_id="openunmix-validate",
        channels=1,
        num_samples=num_samples,
        chunk_index=0,
        total_chunks=1,
        dtype="float32",
        waveform=waveform,
        sample_rate=sample_rate,
    )

    prediction = await openunmix_predict_channels(chunk)

    print("Open-Unmix prediction (per-class score):")
    scores = prediction.predictions[:, 0]
    for idx, score in enumerate(scores):
        label = ci.SoundClassifications(idx).name
        print(f"- {label}: {score:.4f}")


if __name__ == "__main__":
    asyncio.run(run())
