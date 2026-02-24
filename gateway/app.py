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
from librosa import load as librosa_load
from numpy.typing import NDArray

from common.logging_utils import setup_logging
from common.interfaces import AudioChunk, InputAudioFile, OutputAudioFile, JSONResponse


# Constants
BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "static" / "index.html"

# Input data constraints
MAX_STR_LEN = 255
MAX_FILE_SIZE_MB = 10
MB_IN_BYTES = 1024 * 1024
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * MB_IN_BYTES
CHUNK_SIZE = 1024
SAMPLE_RATE = 44100
FUSER_TIMEOUT = 10.0  # seconds
ALLOWED_CONTENT_TYPES = {"audio/wav", "audio/mp3"}

# URLs for downstream services
FFT_URL = "http://localhost:8001/api/fft"
CQT_URL = "http://localhost:8002/api/cqt"
CHROMA_URL = "http://localhost:8003/api/chroma"


# Service objects
app = FastAPI()
logger = setup_logging("gateway")

# Mount static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

pending_events: Dict[str, asyncio.Event] = {}
pending_results: Dict[str, OutputAudioFile] = {}


async def validate_input_file(request_id: str, file: UploadFile) -> InputAudioFile:
    """
    Validates the uploaded audio file and returns an InputAudioFile object.
    
    Args:
        request_id (str): The unique identifier for the request.
        file (UploadFile): The uploaded file to be validated.
        
    Returns:
        InputAudioFile: An InputAudioFile object containing the validated file data.
        
    Raises:
        AssertionError: If the validation fails.
    """
    # Perform validation logic here (e.g., check file type, size, etc.)
    # Raise AssertionError if validation fails
    assert file.content_type in ALLOWED_CONTENT_TYPES, f"Invalid file type. Allowed types: {', '.join(ALLOWED_CONTENT_TYPES)}."
    assert file.filename, "Filename is required."
    assert request_id, "Request ID is required."
    assert len(file.filename) <= MAX_STR_LEN, "Filename is too long."
    assert len(request_id) <= MAX_STR_LEN, "Request ID is too long."
    assert file.size <= MAX_FILE_SIZE_BYTES, f"File size exceeds {MAX_FILE_SIZE_MB} MB limit."
    
    data, Fs = librosa_load(file.file, sr=SAMPLE_RATE, mono=True)
    
    assert data.dtype == np.float32, "Audio data must be in float32 format."
    assert isinstance(data, NDArray[np.float32]), "Audio data must be a numpy array of float32."
    assert data.ndim == 1, "Audio data must be a 1D array. (Mono audio only)"
    assert Fs == SAMPLE_RATE, f"Sample rate must be {SAMPLE_RATE} Hz."
    assert len(data) > 0, "Audio file is too short to be processed."
    
    return InputAudioFile(
        request_id=request_id,
        channels=1,
        num_samples=len(data),
        filename=file.filename,
        num_chunks=int(np.ceil(len(data) / CHUNK_SIZE)),
        dtype=np.float32,
        waveform=data,
        sample_rate=SAMPLE_RATE,
    )
     
async def validate_audio_chunk(request_id: str, chunk: AudioChunk) -> None:
    """
    Validates an AudioChunk object.
    
    Args:
        request_id (str): The unique identifier for the request.
        chunk (AudioChunk): The AudioChunk object to be validated.
        
    Raises:
        AssertionError: If validation fails.
    """
    assert chunk.request_id == request_id, "Chunk request ID does not match."
    assert chunk.chunk_index >= 0, "Chunk index must be non-negative."
    assert chunk.num_chunks > 0, "Number of chunks must be positive."
    assert chunk.channels == 1, "Only mono audio is supported."
    assert chunk.sample_rate == SAMPLE_RATE, f"Sample rate must be {SAMPLE_RATE} Hz."
    assert chunk.dtype == np.float32, "Audio chunk data type must be float32."
    assert isinstance(chunk.waveform, NDArray[np.float32]), "Audio chunk waveform must be a numpy array of float32."
    assert chunk.waveform.ndim == 1, "Audio chunk waveform must be a 1D array."
    assert chunk.waveform.dtype == np.float32, "Audio chunk waveform must be in float32 format."
    assert len(chunk.waveform) <= CHUNK_SIZE, f"Audio chunk size must be at most {CHUNK_SIZE} samples."
        
async def validate_output_file(output_file: OutputAudioFile) -> None:
    """
    Validates an OutputAudioFile object.
    
    Args:
        output_file (OutputAudioFile): The OutputAudioFile object to be validated.
        
    Raises:
        AssertionError: If validation fails.
    """
    assert output_file.request_id, "Request ID is required."
    assert len(output_file.request_id) <= MAX_STR_LEN, "Request ID is too long."
    assert output_file.filename, "Filename is required."
    assert len(output_file.filename) <= MAX_STR_LEN, "Filename is too long."
    assert output_file.channels == 1, "Only mono audio is supported."
    assert output_file.sample_rate == SAMPLE_RATE, f"Sample rate must be {SAMPLE_RATE} Hz."
    assert output_file.dtype == np.float32, "Audio data type must be float32."
    assert isinstance(output_file.waveform, NDArray[np.float32]), "Audio waveform must be a numpy array of float32."
    assert output_file.waveform.ndim == 1, "Audio waveform must be a 1D array."
    assert output_file.waveform.dtype == np.float32, "Audio waveform must be in float32 format."
    assert len(output_file.waveform) > 0, "Output audio file is too short to be valid."
        
async def send_to_transforms(chunk: AudioChunk) -> None:
    """
    Sends the initial payload to each of the transform services (FFT, CQT, Chroma).
    
    Args:
        chunk (AudioChunk): The audio chunk to be processed.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await asyncio.gather(
                client.post(FFT_URL, json=chunk.model_dump()),
                client.post(CQT_URL, json=chunk.model_dump()),
                client.post(CHROMA_URL, json=chunk.model_dump()),
            )
    except Exception:
        logger.exception("Request failed")
        pending_events.pop(chunk.request_id, None)
        pending_results.pop(chunk.request_id, None)

async def await_fuser(request_id: str, timeout: float = FUSER_TIMEOUT) -> OutputAudioFile | None:
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
        audio_file = validate_input_file(request_id, file)
    except AssertionError as exc:
        logger.error(f"File validation failed: {exc}")
        return {"message": f"FAIL (File validation error): {exc}"}

    event = asyncio.Event()
    pending_events[request_id] = event

    async for chunk in audio_file:
        try:
            validate_audio_chunk(request_id, chunk)
        except AssertionError as exc:
            logger.error(f"Chunk validation failed: {exc}")
            return {"message": f"FAIL (Chunk validation error): {exc}"}
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
async def final(output_file: OutputAudioFile) -> JSONResponse:
    """
    Endpoing to receive the final fused audio file from the channel fuser and return the response to the web application.
    
    Args:
        output_file: The OutputAudioFile object containing the final fused audio data.
        
    Returns:
        A JSON response containing the final result message.
    """
    
    logger.info(f"Final response received: {output_file.filename} with request_id: {output_file.request_id}")
    pending_results[output_file.request_id] = output_file
    
    try:
        validate_output_file(output_file)
        event = pending_events.get(output_file.request_id)
        if event:
            event.set()
    except AssertionError as exc:
        logger.error(f"Output validation failed: {exc}")
        ret_json = {
            "status": 400,
            "message": f"FAIL (Output validation error): {exc}"
        }
    except KeyError:
        logger.error(f"No pending event found for request_id: {output_file.request_id}")
        ret_json = {
            "status": 404,
            "message": f"FAIL: No pending request found for request_id: {output_file.request_id}"
        }
        
    return ret_json
