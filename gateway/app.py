
from pathlib import Path
import asyncio

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from common.logging_utils import setup_logging
from common.interfaces import AlertPayload


# Constants
BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "static" / "index.html"
TMP_DIR = BASE_DIR / ".tmp"
FFT_URL = "http://localhost:8001/api/fft"
CQT_URL = "http://localhost:8002/api/cqt"
CHROMA_URL = "http://localhost:8003/api/chroma"


# Service objects
app = FastAPI()
logger = setup_logging("gateway")

# Mount static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

pending_events: dict[str, asyncio.Event] = {}
pending_results: dict[str, str] = {}

@app.get("/")
async def root():
    return HTMLResponse(content=INDEX_HTML.read_text(encoding="utf-8"))

@app.post("/api/alert")
async def alert(
    request_id: str = Form(...),
    file: UploadFile = File(...),
):
    TMP_DIR.mkdir(exist_ok=True)
    safe_name = Path(file.filename or "upload.bin").name
    target_path = TMP_DIR / f"{request_id}_{safe_name}"
    contents = await file.read()
    target_path.write_bytes(contents)

    logger.info(f"Received file: {safe_name}")

    event = asyncio.Event()
    pending_events[request_id] = event

    base_trace = ["gateway:ACK"]
    fft_payload = AlertPayload(
        request_id=request_id,
        message=str(target_path),
        trace=base_trace,
        source="gateway",
    )
    cqt_payload = fft_payload.model_copy(update={"source": "gateway"})
    chroma_payload = fft_payload.model_copy(update={"source": "gateway"})

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await asyncio.gather(
                client.post(FFT_URL, json=fft_payload.model_dump()),
                client.post(CQT_URL, json=cqt_payload.model_dump()),
                client.post(CHROMA_URL, json=chroma_payload.model_dump()),
            )
    except Exception as exc:
        logger.exception("Request failed")
        pending_events.pop(request_id, None)
        pending_results.pop(request_id, None)
        return {"message": f"Request failed: {exc}"}

    try:
        await asyncio.wait_for(event.wait(), timeout=10.0)
    except asyncio.TimeoutError:
        pending_events.pop(request_id, None)
        pending_results.pop(request_id, None)
        return {"message": "Timed out waiting for channel fuser."}

    result = pending_results.pop(request_id, "No response from channel fuser.")
    pending_events.pop(request_id, None)
    return {"message": result}

@app.post("/api/final")
async def final(payload: AlertPayload):
    logger.info(f"Final response received: {payload.message}")
    result = payload.message
    if payload.trace:
        result = f"{payload.message}\n\nTrace:\n" + "\n".join(payload.trace)
    pending_results[payload.request_id] = result
    event = pending_events.get(payload.request_id)
    if event:
        event.set()
    return {"message": "ACK"}
