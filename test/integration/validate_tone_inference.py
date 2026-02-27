import asyncio
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from librosa import load as librosa_load

import common.constants as cc
import common.interfaces as ci
from prediction.tone_identifier import build_mel_spectrogram, compute_tone_prediction


TEST_FILE = Path("test/resources/test-blues.mp3")
RESULTS_DIR = Path("gateway/results")


async def run() -> None:
    """
    Load a test audio file, build ToneInput payloads for the first 8 chunks,
    and print AST-based tone predictions to the console.
    """
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Missing test file: {TEST_FILE}")

    data, sample_rate = librosa_load(TEST_FILE, sr=cc.SAMPLE_RATE, mono=True)
    audio_file = ci.InputAudioFile(
        request_id="tone-validate",
        filename=TEST_FILE.name,
        channels=1,
        num_samples=len(data),
        num_chunks=int(np.ceil(len(data) / cc.CHUNK_SIZE)),
        dtype="float32",
        waveform=data.astype(np.float32),
        sample_rate=sample_rate,
    )

    collected: list[ci.AudioChunk] = []
    idx = 0
    async for chunk in audio_file:
        if idx < 8:
            idx += 1
            continue
        collected.append(chunk)
        if idx >= 15:
            break
        idx += 1

    samples = np.concatenate([c.waveform for c in collected], axis=0).astype(np.float32)
    mel = await build_mel_spectrogram(samples, sample_rate, audio_file.request_id)
    prediction = await compute_tone_prediction(samples, mel)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mel_path = RESULTS_DIR / f"{audio_file.request_id}-mel.npy"
    img_path = RESULTS_DIR / f"{audio_file.request_id}-mel.png"

    np.save(mel_path, mel.mel_spectrogram)
    mel_db = librosa.power_to_db(mel.mel_spectrogram, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        hop_length=mel.hop_length,
        x_axis="time",
        y_axis="mel",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close()

    print(f"Saved mel spectrogram: {mel_path}")
    print(f"Saved mel spectrogram image: {img_path}")

    print("Tone prediction:")
    for idx, (class_id, confidence) in enumerate(prediction.class_probabilities):
        try:
            label = ci.SoundClassifications(int(class_id)).name
        except Exception:
            label = f"class_{idx}"
        print(f"- {label}: {confidence:.4f}")


if __name__ == "__main__":
    asyncio.run(run())
