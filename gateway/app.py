"""
Primary application for the "Gateway" Service in the audio processing pipeline.
This is the primary contact point for the web application and is responsible for:
- Receiving audio file uploads from the web application.
- Chunking the audio file into 1024-sample segments (if necessary/possible).
- Forwarding the audio chunks to the three transform services (FFT, CQT, Chroma).
- Awaiting the results from the channel fuser and returning the final response to the web application.

The Gateway app adheres to the following Contract:

Boundary: ** Web Application -> Gateway **
Uses `InputFile` dataclass for the incoming file

Boundary: ** (Internal) **
Uses `InputAudioFile` to convert and validate the input audio file
Uses `AudioChunk` dataclass to split the audio file into 1024-sample segments (if necessary/possible)

Boundary: ** Gateway -> Transforms (FFT, CQT, Chroma) **
Uses `FFTChunk`, `CQTChunk`, and `ChromaChunk` dataclasses for the outgoing audio chunks (depending on the transform)

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
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import common.constants as cc
import common.interfaces as ci
from common.logging_utils import setup_logging
from librosa import load as librosa_load


# Constants
BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "static" / "index.html"

# Service objects
app = FastAPI()
logger = setup_logging("gateway")
pending_events: Dict[str, asyncio.Event] = {}
pending_results: Dict[str, ci.OutputAudioFile] = {}

# Mount static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


async def create_input_file(request_id: str, file: UploadFile) -> ci.InputAudioFile:
    assert request_id, "Request ID is required."
    assert len(request_id) <= cc.MAX_STR_LEN, "Request ID is too long."
    assert file.filename, "Filename is required."
    assert len(file.filename) <= cc.MAX_STR_LEN, "Filename is too long."
    assert file.content_type in cc.ALLOWED_CONTENT_TYPES, (
        f"Invalid file type. Allowed types: {', '.join(cc.ALLOWED_CONTENT_TYPES)}."
    )

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


async def send_to_transforms(chunk: ci.AudioChunk) -> None:
    """
    Sends the initial payload to each of the transform services (FFT, CQT, Chroma).
    
    Args:
        chunk (AudioChunk): The audio chunk to be processed.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await asyncio.gather(
                client.post(cc.FFT_URL, json=chunk.model_dump(mode="json")),
                client.post(cc.CQT_URL, json=chunk.model_dump(mode="json")),
                client.post(cc.CHROMA_URL, json=chunk.model_dump(mode="json")),
            )
    except Exception:
        logger.exception("Request failed")
        pending_events.pop(chunk.request_id, None)
        pending_results.pop(chunk.request_id, None)

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
) -> ci.JSONResponse:
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
    except AssertionError as exc:
        logger.error(f"File validation failed: {exc}")
        return {"message": f"FAIL (File validation error): {exc}"}

    event = asyncio.Event()
    pending_events[request_id] = event

    async for chunk in audio_file:
        if chunk.request_id != request_id:
            logger.error("Chunk validation failed: Chunk request ID does not match.")
            return {"message": "FAIL (Chunk validation error): Chunk request ID does not match."}
        logger.info(f"Processing chunk {chunk.chunk_index}/{audio_file.num_chunks} for request_id: {request_id}")
        await send_to_transforms(chunk)

    result = await await_fuser(request_id)
    if result is None:
        logger.error(f"Channel fuser did not respond within timeout for request_id: {request_id}")
        ret_json = {
            "status": 504,
            "message": "FAIL: Channel fuser timeout."
        }
    else:
        logger.info(f"Received final result from channel fuser for request_id: {request_id}")
        ret_json = {
            "status": 200,
            "message": result.model_dump_json()
        }
    
    pending_events.pop(request_id, None)
    return ret_json

@app.post("/api/final")
async def final(output_file: ci.OutputAudioFile) -> ci.JSONResponse:
    """
    Endpoing to receive the final fused audio file from the channel fuser and return the response to the web application.
    
    Args:
        output_file: The OutputAudioFile object containing the final fused audio data.
        
    Returns:
        A JSON response containing the final result message.
    """
    try:
        output_file.validate_contents()
    except AssertionError as exc:
        logger.error(f"Output file validation failed: {exc}")
        return {
            "status": 400,
            "message": f"FAIL (Output file validation error): {exc}"
        }
    
    logger.info(f"Final response received: {output_file.filename} with request_id: {output_file.request_id}")
    pending_results[output_file.request_id] = output_file

    event = pending_events.get(output_file.request_id)
    if event:
        event.set()
        return {"status": 200, "message": "ACK"}

    logger.error(f"No pending event found for request_id: {output_file.request_id}")
    return {
        "status": 404,
        "message": f"FAIL: No pending request found for request_id: {output_file.request_id}",
    }
