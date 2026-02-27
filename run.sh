#!/usr/bin/env bash
# run.sh — Start all pipeline services or run the test suite.
#
# Usage:
#   ./run.sh services   — start all services (each in background)
#   ./run.sh unit-tests       — run pytest
#   ./run.sh install    — pip install in editable mode with dev deps

set -euo pipefail

case "${1:-help}" in
  install)
    pip install -e ".[dev]"
    ;;

  services)
    echo "Starting all services …"
    python -m uvicorn transforms.fft:app                --host 0.0.0.0 --port 8001 --reload &
    python -m uvicorn transforms.cqt:app                --host 0.0.0.0 --port 8002 --reload &
    python -m uvicorn prediction.tone_identifier:app    --host 0.0.0.0 --port 8004 --reload &
    python -m uvicorn prediction.channel_predictor:app  --host 0.0.0.0 --port 8005 --reload &
    python -m uvicorn prediction.channel_fuser:app      --host 0.0.0.0 --port 8006 --reload &
    python -m uvicorn gateway.app:app                   --host 0.0.0.0 --port 8000 --reload &
    echo "All services started. Gateway at http://localhost:8000"
    echo "Logs: tail -f logs/pipeline.log"
    wait
    ;;

  unit-tests)
    python -m pytest test/unit -v
    ;;

  integration-tests)
    python -m pytest test/integration -v
    ;;

  *)
    echo "Usage: ./run.sh {install|services|unit-tests}"
    exit 1
    ;;
esac
