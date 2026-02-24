
import asyncio
import httpx
from fastapi import FastAPI

from common.logging_utils import setup_logging
from common.interfaces import AlertPayload

# Service objects
app = FastAPI()
logger = setup_logging("prediction-CHANNEL_PREDICTOR")

CHANNEL_FUSER_URL = "http://localhost:8006/api/channel_fuser"
EXPECTED_SOURCES = {"fft", "cqt", "chroma", "tone_identifier"}

_lock = asyncio.Lock()
_requests: dict[str, dict[str, object]] = {}

@app.post("/api/channel_predictor")
async def channel_predictor(payload: AlertPayload):
    logger.info(f"Received alert: {payload.message}; Source: {payload.source}")
    async with _lock:
        entry = _requests.setdefault(
            payload.request_id,
            {"message": payload.message, "trace": set(), "sources": set()},
        )
        entry["trace"].update(payload.trace)
        entry["sources"].add(payload.source)
        entry["trace"].add("channel_predictor:ACK")

        sources = entry["sources"]
        trace = entry["trace"]
        message = entry["message"]

        if not EXPECTED_SOURCES.issubset(sources):
            return {"message": "ACK"}

        _requests.pop(payload.request_id, None)

    fused_payload = AlertPayload(
        request_id=payload.request_id,
        message=message,
        trace=sorted(trace),
        source="channel_predictor",
    )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(CHANNEL_FUSER_URL, json=fused_payload.model_dump())
    except Exception as exc:
        logger.exception("Forwarding failed")
        return {"message": f"FAIL: channel_predictor forwarding error: {exc}"}

    return {"message": "ACK"}
