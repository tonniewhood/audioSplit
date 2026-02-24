# audio-separation

Basic project scaffold for a Python-based audio separation system with a FastAPI gateway and placeholder service packages.

## Current repository layout

```text
audioSplit/
├── .gitignore
├── pyproject.toml
├── README.md
├── run.sh
├── common/                     # currently empty
├── gateway/
│   ├── app.py                  # FastAPI app entrypoint
│   ├── static/
│   │   └── styles.css
│   └── templates/
│       └── index.html
├── prediction/                 # currently empty
├── transforms/                 # currently empty
└── test/
	├── integration/            # currently empty
	└── unit/                   # currently empty
```

## Python package metadata

Defined in `pyproject.toml`:

- Package name: `audio-separation`
- Version: `0.1.0`
- Python: `>=3.10`
- Included package namespaces: `common*`, `transforms*`, `gateway*`, `prediction*`

Primary dependencies currently listed:

- `fastapi`
- `uvicorn[standard]`
- `httpx`
- `pydantic`
- `numpy`
- `librosa`
- `jinja2`
- `python-multipart`

Dev dependencies:

- `pytest`
- `pytest-asyncio`

## Existing scripts

`run.sh` currently supports:

- `./run.sh install` — install package in editable mode with dev dependencies
- `./run.sh services` — start multiple uvicorn services (as defined in script)
- `./run.sh test` — run unit tests via pytest

## Notes

- The `gateway` package is the only package with implemented files right now.
- `common`, `prediction`, `transforms`, and test subfolders are present as placeholders.
