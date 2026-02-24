
import httpx
from fastapi import FastAPI

from common.logging_utils import setup_logging
from common.interfaces import AlertPayload

# Service objects
app = FastAPI()
logger = setup_logging("transforms-FFT")

TONE_IDENTIFIER_URL = "http://localhost:8004/api/tone_identifier"
CHANNEL_PREDICTOR_URL = "http://localhost:8005/api/channel_predictor"

@app.post("/api/fft")
async def fft(payload: AlertPayload):
    logger.info(f"Received alert: {payload.message}; Source: {payload.source}")
    trace = payload.trace + ["fft:ACK"]
    fft_payload = AlertPayload(
        request_id=payload.request_id,
        message=payload.message,
        trace=trace,
        source="fft",
    )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(TONE_IDENTIFIER_URL, json=fft_payload.model_dump())
            await client.post(CHANNEL_PREDICTOR_URL, json=fft_payload.model_dump())
    except Exception as exc:
        logger.exception("Forwarding failed")
        return {"message": f"FAIL: fft forwarding error: {exc}"}

    return {"message": "ACK"}
