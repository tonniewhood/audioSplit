# audio-separation

Distributed audio separation pipeline with a FastAPI gateway and multiple services. The system accepts an audio file, chunks it, performs FFT and tone inference, runs Open-Unmix separation on raw audio, and fuses outputs into a multi-channel WAV.

## Current architecture (high level)

- **Gateway**: upload + validation + chunking + fan-out
- **Transforms**
  - FFT: computes FFT features, forwards ToneInput to tone identifier
  - CQT: placeholder transform (not used in prediction path)
- **Prediction**
  - Tone Identifier: builds mel-spectrogram + AST tone prediction
  - Channel Predictor: Open-Unmix separation (temporal/raw audio)
  - Channel Fuser: merges predicted chunks and emits final WAV

## Repository layout

```text
audioSplit/
├── Project-Part-2.md
├── pyproject.toml
├── README.md
├── run.sh
├── common/
│   ├── constants.py
│   ├── interfaces.py
│   └── logging_utils.py
├── gateway/
│   ├── app.py
│   ├── results/
│   └── static/
├── models/                      # model cache (AST, Open-Unmix)
├── prediction/
│   ├── channel_fuser.py
│   ├── channel_predictor.py
│   ├── openunmix_predict.py
│   ├── temporal_predict.py
│   └── tone_identifier.py
├── transforms/
│   ├── cqt.py
│   └── fft.py
└── test/
    ├── integration/
    │   ├── validate_openunmix_inference.py
    │   └── validate_tone_inference.py
    ├── resources/
    │   ├── test-2048.wav
    │   ├── test-440hz-10s.wav
    │   ├── test-blues.mp3
    │   └── test-blues-10s.wav
    └── unit/
```

## Install

Requires Python 3.10+.

```bash
./run.sh install
```

This installs:
- FastAPI + Uvicorn
- Librosa + NumPy
- PyTorch + TorchAudio
- Transformers (AST)
- Matplotlib (for spectrogram export)

## Run services

```bash
./run.sh services
```

Services started:
- Gateway: `http://localhost:8000`
- FFT: `:8001`
- CQT: `:8002`
- Tone Identifier: `:8004`
- Channel Predictor: `:8005`
- Channel Fuser: `:8006`

## Tests

```bash
./run.sh unit-tests
./run.sh integration-tests
```

Integration helpers:
- `python test/integration/validate_tone_inference.py`
- `python test/integration/validate_openunmix_inference.py`

## Models & caching

Models are cached under `models/`:
- AST: `MIT/ast-finetuned-audioset-10-10-0.4593`
- Open-Unmix: `umxhq`

The cache directory is controlled via constants in `common/constants.py`.

## Outputs

Final WAVs are written to:

```
gateway/results/
```

The gateway also exposes:
- `/api/status/{request_id}`
- `/api/result/{request_id}`

## Key configuration

Defined in `common/constants.py`:
- `CHUNK_SIZE = 4096`
- `SAMPLE_RATE = 44100`
- `MEL_N_MELS = 128`
- `MEL_HOP_LENGTH = CHUNK_SIZE - SPEC_OVERLAP`
