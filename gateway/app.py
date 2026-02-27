"""
Primary application for the "Gateway" Service in the audio processing pipeline.
This is the primary contact point for the web application and is responsible for:
- Receiving audio file uploads from the web application.
- Chunking the audio file into CHUNK_SIZE segments (if necessary/possible).
- Forwarding the audio chunks to the transform services (FFT, CQT) and temporal predictor.
- Awaiting the results from the channel fuser and returning the final response to the web application.

The Gateway app adheres to the following Contract:

Boundary: ** Web Application -> Gateway **
Uses `InputFile` dataclass for the incoming file

Boundary: ** (Internal) **
Uses `InputAudioFile` to convert and validate the input audio file
Uses `AudioChunk` dataclass to split the audio file into CHUNK_SIZE segments (if necessary/possible)

Boundary: ** Gateway -> Transforms (FFT, CQT) / Temporal Predictor **
Uses `FFTChunk`, `CQTChunk`, and `AudioChunk` dataclasses for the outgoing audio chunks (depending on the target)

Boundary: ** Fusion -> Gateway **
Uses `OutputAudioFile` dataclass for the final fused audio file response from the channel fuser

Boundary: ** Gateway -> Web Application **
Uses `OutputAudioFile` dataclass for the final response back to the web application

Inputs:
- Audio file uploads from the web application (via multipart/form-data)
- Final fused audio file from the channel fuser (via JSON payload)
- Acknowledgments from transform services (via JSON payloads)
Outputs:
- Forwarding of audio chunks to transform services (via JSON payloads)
- Final response back to the web application (via JSON payload)
- Acknowledgments to transform services (via JSON payloads)
- Logging of all received and sent messages for traceability
- Logging of all abnormal/exceptional events for debugging and monitoring purposes

"""

import asyncio
from pathlib import Path
from typing import Dict

import httpx
import numpy as np
import wave
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
import common.constants as cc
import common.interfaces as ci
from common.logging_utils import setup_logging
from librosa import load as librosa_load


# Constants
BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "static" / "index.html"
RESULT_DIR = BASE_DIR / "results"

# Service objects
app = FastAPI()
logger = setup_logging("gateway")
pending_events: Dict[str, asyncio.Event] = {}
pending_results: Dict[str, ci.OutputAudioFile] = {}
request_status: Dict[str, str] = {}
request_errors: Dict[str, str] = {}
request_filenames: Dict[str, str] = {}
result_paths: Dict[str, Path] = {}
_cancelled_requests: set[str] = set()

CANCEL_URLS = [
    cc.FFT_CANCEL_URL,
    cc.CQT_CANCEL_URL,
    cc.TONE_IDENTIFIER_CANCEL_URL,
    cc.CHANNEL_PREDICTOR_CANCEL_URL,
    cc.CHANNEL_FUSER_CANCEL_URL,
]

# Mount static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


async def create_input_file(request_id: str, file: UploadFile) -> ci.InputAudioFile:
    assert request_id, "Request ID is required."
    assert len(request_id) <= cc.MAX_STR_LEN, "Request ID is too long."
    assert file.filename, "Filename is required."
    assert len(file.filename) <= cc.MAX_STR_LEN, "Filename is too long."
    assert (
        file.content_type in cc.ALLOWED_CONTENT_TYPES
    ), f"Invalid file type. Allowed types: {', '.join(cc.ALLOWED_CONTENT_TYPES)}."

    request_filenames[request_id] = file.filename

    data, sample_rate = librosa_load(file.file, sr=cc.SAMPLE_RATE, mono=True)

    return ci.InputAudioFile(
        request_id=request_id,
        channels=1,
        num_samples=len(data),
        filename=file.filename,
        num_chunks=int(np.ceil(len(data) / cc.CHUNK_SIZE)),
        dtype="float32",
        waveform=data,
        sample_rate=sample_rate,
    )


async def send_to_transforms(chunk: ci.AudioChunk) -> bool:
    """
    Sends the initial payload to the FFT transform and temporal predictor.

    Args:
        chunk (AudioChunk): The audio chunk to be processed.

    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    try:
        async with httpx.AsyncClient(timeout=cc.HTTP_TIMEOUT) as client:
            await asyncio.gather(
                client.post(cc.FFT_URL, json=chunk.model_dump(mode="json")),
                client.post(cc.build_predictor_url("temporal"), json=chunk.model_dump(mode="json")),
            )
    except (httpx.ReadTimeout, httpx.ReadError):
        logger.warning(f"Downstream timeout for chunk: {chunk.chunk_index} of request_id: {chunk.request_id}")
        return False
    except Exception:
        logger.exception("Request failed")
        pending_events.pop(chunk.request_id, None)
        pending_results.pop(chunk.request_id, None)
        return False

    return True


async def await_fuser(request_id: str, timeout: float = cc.FUSER_TIMEOUT) -> ci.OutputAudioFile | None:
    """
    Waits for the final result from the channel fuser for a given request ID.

    Args:
        request_id (str): The unique identifier for the request.
        timeout (float): Maximum time to wait for the result in seconds.

    Returns:
        OutputAudioFile | None: The final fused audio file if received within the timeout, otherwise None.
    """
    event = pending_events.get(request_id)
    if event is None:
        event = asyncio.Event()
        pending_events[request_id] = event
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return pending_results.pop(request_id, None)
    except asyncio.TimeoutError:
        pending_events.pop(request_id, None)
        return pending_results.pop(request_id, None)


@app.get("/")
async def root():
    """Serves the main HTML page for the web application."""
    return HTMLResponse(content=INDEX_HTML.read_text(encoding="utf-8"))


@app.post("/api/alert")
async def alert(
    request_id: str = Form(...),
    file: UploadFile = File(...),
) -> JSONResponse:
    """
    Endpoint to receive the uploaded audio file from the web application, validate it, chunk it, and forward the chunks to the transform services.

    Args:
        request_id (str): The unique identifier for the request, provided as a form field.
        file (UploadFile): The uploaded audio file, provided as a file field.

    Returns:
        RestResponse: A response object containing the result of the operation.
    """
    # target_path = await write_temp_file(request_id, file)
    logger.info(f"Received file: {file.filename} with request_id: {request_id}")

    try:
        audio_file = await create_input_file(request_id, file)
    except (AssertionError, ValidationError) as exc:
        logger.error(f"File validation failed: {exc}")
        return JSONResponse(
            status_code=400,
            content={"status": 400, "message": f"Failed to validate file. {exc}"},
        )
    except Exception as exc:
        logger.exception(f"Failed to process file for request_id: {request_id}: {exc}")
        return JSONResponse(
            status_code=400,
            content={"status": 400, "message": "Internal service error"},
        )

    request_status[request_id] = "pending"

    async for chunk in audio_file:
        if request_id in _cancelled_requests or request_status.get(request_id) == "error":
            logger.warning(f"Stopping processing for request_id: {request_id} due to error/cancel")
            return JSONResponse(
                status_code=409,
                content={"status": 409, "message": "CANCELLED"},
            )
        if chunk.request_id != request_id:
            logger.error("Chunk validation failed: Chunk request ID does not match.")
            return JSONResponse(
                status_code=500,
                content={"status": 500, "message": "Internal service error"},
            )
        logger.info(f"Processing chunk {chunk.chunk_index}/{audio_file.num_chunks} for request_id: {request_id}")
        if not await send_to_transforms(chunk):
            return JSONResponse(
                status_code=504,
                content={"status": 504, "message": "Internal service error"},
            )

    return JSONResponse(status_code=200, content={"status": 200, "message": "ACK"})


@app.post("/api/final")
async def final(output_file: ci.OutputAudioFile) -> JSONResponse:
    """
    Endpoing to receive the final fused audio file from the channel fuser and return the response to the web application.

    Args:
        output_file: The OutputAudioFile object containing the final fused audio data.

    Returns:
        A JSON response containing the final result message.
    """
    try:
        output_file.validate_contents()
    except (AssertionError, ValidationError) as exc:
        logger.error(f"Output file validation failed: {exc}")
        return JSONResponse(
            status_code=400,
            content={"status": 400, "message": f"FAIL (Output file validation error): {exc}"},
        )

    logger.info(f"Final response received with request_id: {output_file.request_id}")
    pending_results[output_file.request_id] = output_file

    RESULT_DIR.mkdir(exist_ok=True)
    out_path = RESULT_DIR / f"{request_filenames.get(output_file.request_id, output_file.request_id)}-split.wav"

    waveform = output_file.waveform
    if waveform.ndim == 1:
        samples = waveform
        channels = 1
    else:
        channels = output_file.channels
        samples = waveform.T.reshape(-1)

    samples = np.clip(samples, -1.0, 1.0)
    pcm16 = (samples * 32767.0).astype(np.int16)

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(cc.SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())

    result_paths[output_file.request_id] = out_path
    request_status[output_file.request_id] = "done"

    return JSONResponse(status_code=200, content={"status": 200, "message": "ACK; Final result received"})


@app.post("/api/error")
async def error(payload: ci.ErrorPayload) -> JSONResponse:
    request_errors[payload.request_id] = f"{payload.source}: {payload.message}"
    request_status[payload.request_id] = "error"
    _cancelled_requests.add(payload.request_id)
    try:
        async with httpx.AsyncClient(timeout=cc.HTTP_ERROR_TIMEOUT) as client:
            await asyncio.gather(
                *[client.post(f"{url}/{payload.request_id}") for url in CANCEL_URLS],
                return_exceptions=True,
            )
    except httpx.ReadTimeout:
        logger.warning(f"Cancel broadcast timeout for request_id {payload.request_id}")
    except Exception:
        logger.exception("Failed to broadcast cancel requests")
    return JSONResponse(status_code=200, content={"status": 200, "message": "ACK"})


@app.get("/api/status/{request_id}")
async def status(request_id: str):
    status = request_status.get(request_id)
    if status == "pending":
        return JSONResponse(status_code=200, content={"status": "pending"})
    if status == "done":
        return JSONResponse(status_code=201, content={"status": "done"})
    if status == "error":
        return JSONResponse(status_code=500, content={"status": "error", "message": "internal service error"})
    return JSONResponse(status_code=404, content={"status": "unknown request id"})


@app.get("/api/result/{request_id}")
async def result(request_id: str):
    path = result_paths.get(request_id)
    if not path or not path.exists():
        return JSONResponse(status_code=404, content={"status": "not_found"})
    response = FileResponse(path, media_type="audio/wav", filename=path.name)
    response.headers["X-Result-Filename"] = path.name
    return response
