"""
Houses the project constants to keep them organized and maintainable.
Grouped by purpose for easy discovery.
"""

# Validation limits
MAX_STR_LEN = 255
NUM_PITCH_CLASSES = 12
MAX_INPUT_BYTES = 10 * 1024 * 1024
CHUNK_SIZE = 1024
MAX_CHUNK_BUFFER_SIZE = 8
SPEC_OVERLAP = 256
SPEC_FRAMES = 1 + (MAX_CHUNK_BUFFER_SIZE * CHUNK_SIZE - CHUNK_SIZE) // (CHUNK_SIZE - SPEC_OVERLAP)

# Audio defaults
SAMPLE_RATE = 44100

# Prediction info
NUM_PREDICTORS = 3

# Gateway timing
FUSER_TIMEOUT = 3.0

# HTTP client timeouts
HTTP_TIMEOUT = 20.0
HTTP_ERROR_TIMEOUT = 10.0

# Upload constraints
ALLOWED_CONTENT_TYPES = {"audio/wav", "audio/mp3"}

# Service ports
GATEWAY_PORT = 8000
FFT_PORT = 8001
CQT_PORT = 8002
CHROMA_PORT = 8003
TONE_IDENTIFIER_PORT = 8004
CHANNEL_PREDICTOR_PORT = 8005
CHANNEL_FUSER_PORT = 8006

# Service URLs
GATEWAY_FINAL_URL = f"http://localhost:{GATEWAY_PORT}/api/final"
GATEWAY_ERROR_URL = f"http://localhost:{GATEWAY_PORT}/api/error"
FFT_URL = f"http://localhost:{FFT_PORT}/api/fft"
CQT_URL = f"http://localhost:{CQT_PORT}/api/cqt"
CHROMA_URL = f"http://localhost:{CHROMA_PORT}/api/chroma"
TONE_IDENTIFIER_URL = f"http://localhost:{TONE_IDENTIFIER_PORT}/api/tone_identifier"
CHANNEL_PREDICTOR_URL = "http://localhost:{CHANNEL_PREDICTOR_PORT}/api/{source}/channel_predictor"
CHANNEL_FUSER_URL = f"http://localhost:{CHANNEL_FUSER_PORT}/api/channel_fuser"

# Cancel endpoints
GATEWAY_ERROR_URL = f"http://localhost:{GATEWAY_PORT}/api/error"
FFT_CANCEL_URL = f"http://localhost:{FFT_PORT}/api/cancel"
CQT_CANCEL_URL = f"http://localhost:{CQT_PORT}/api/cancel"
CHROMA_CANCEL_URL = f"http://localhost:{CHROMA_PORT}/api/cancel"
TONE_IDENTIFIER_CANCEL_URL = f"http://localhost:{TONE_IDENTIFIER_PORT}/api/cancel"
CHANNEL_PREDICTOR_CANCEL_URL = f"http://localhost:{CHANNEL_PREDICTOR_PORT}/api/cancel"
CHANNEL_FUSER_CANCEL_URL = f"http://localhost:{CHANNEL_FUSER_PORT}/api/cancel"

def build_predictor_url(source: str) -> str:
    """
    Build the appropriate URL for the given source.
    
    Args:
        source (str): The source of the prediction (e.g., "fft", "cqt", "chroma").
    
    Returns:
        str: The URL to send the prediction to.
    """
    return CHANNEL_PREDICTOR_URL.format(source=source, CHANNEL_PREDICTOR_PORT=CHANNEL_PREDICTOR_PORT)
