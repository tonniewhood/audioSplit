"""
Houses the project constants to keep them organized and maintainable.
Grouped by purpose for easy discovery.
"""

# Validation limits
MAX_STR_LEN = 255
MAX_CHUNK_SIZE = 1024
NUM_PITCH_CLASSES = 12
MAX_INPUT_BYTES = 10 * 1024 * 1024

# Audio defaults
SAMPLE_RATE = 44100

# Prediction info
NUM_PREDICTORS = 3

# Gateway timing
FUSER_TIMEOUT = 10.0

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
FFT_URL = f"http://localhost:{FFT_PORT}/api/fft"
CQT_URL = f"http://localhost:{CQT_PORT}/api/cqt"
CHROMA_URL = f"http://localhost:{CHROMA_PORT}/api/chroma"
TONE_IDENTIFIER_URL = f"http://localhost:{TONE_IDENTIFIER_PORT}/api/tone_identifier"
CHANNEL_PREDICTOR_URL = f"http://localhost:{CHANNEL_PREDICTOR_PORT}/api/channel_predictor"
CHANNEL_FUSER_URL = f"http://localhost:{CHANNEL_FUSER_PORT}/api/channel_fuser"
