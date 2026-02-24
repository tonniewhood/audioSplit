
from enum import Enum
from typing import Dict, List, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field


# Basic Alises
JSONResponse = Dict[str, str | int]

# Useful Enums
class Pitches(Enum):
    """Represents the 12 pitch classes in Western music. The pitch classes are defined as follows:
    """
    C = 0
    C_SHARP = 1
    D = 2
    D_SHARP = 3
    E = 4
    F = 5
    F_SHARP = 6
    G = 7
    G_SHARP = 8
    A = 9
    A_SHARP = 10
    B = 11
    
    
class SoundClassifications(Enum):
    """Represents the sound classificiations that are current supported
    """
    VOCAL = 0
    GUITAR = 1
    BASS = 2
    DRUMS = 3
    PIANO = 4

# Contract Dataclasses
    
class InputAudioFile(BaseModel):
    """
    Represents an input audio file after preprocessing. The audio file SHALL adhere to the following specifications:
    
    Boundary: (Preprocessing -> Chunking) | (Fusion -> Gateway)
    Version: input_audio_file_v1
    
    Constraints:
        - File must be a valid WAV format
        - Sample rate must be 44.1kHz
        - Bit depth must be 32-bit
        - Channels must be mono
        - File size must not exceed 20MB
    
    Data Fields:
        - request_id (str): Unique identifier for the request
        - channels (int): Number of audio channels (must be 1 for mono)
        - num_samples (int): Total number of samples in the audio file
        - num_chunks (int): Total number of chunks the audio file will be split into
        - dtype (np.float32): Data type of the audio samples (must be float32)
        - waveform (NDArray[np.float32]): 1D array containing the audio samples
        - sample_rate (int): Sample rate of the audio (must be 44100 Hz)
    """
    
    request_id: str
    channels: int
    num_samples: int
    num_chunks: int
    dtype: np.float32
    waveform: NDArray[np.float32]
    sample_rate: int = 44100
    version: Literal["input_audio_file_v1"] = "input_audio_file_v1"
    
    def __aiter__(self):
        """Async generator method to yield AudioChunk objects from the input audio file."""
        total_chunks = self.num_chunks
        for chunk_index in range(total_chunks):
            start_sample = chunk_index * 1024
            end_sample = min(start_sample + 1024, self.num_samples)
            chunk_data = self.waveform[start_sample:end_sample]
            yield AudioChunk(
                request_id=self.request_id,
                channels=self.channels,
                num_samples=len(chunk_data),
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                dtype=self.dtype,
                waveform=chunk_data,
                sample_rate=self.sample_rate,
            )
        
class OutputAudioFile(BaseModel):
    """
    Represents an output audio file after fusion. The audio file SHALL adhere to the following specifications:
    
    Boundary: Channel Fuser -> Gateway
    Version: output_audio_file_v1
    
    Constraints:
        - File must be a valid WAV format
        - Sample rate must be 44.1kHz
        - Bit depth must be 32-bit
        - Must have <= 5 channels (one for each active SoundClassification)
        - File size must not exceed 20MB
    
    Data Fields:
        - request_id (str): Unique identifier for the request
        - channels (int): Number of audio channels
        - num_samples (int): Total number of samples in the audio file
        - dtype (np.float32): Data type of the audio samples (must be float32)
        - waveform (NDArray[np.float32]): Array containing the audio samples
        - sample_rate (int): Sample rate of the audio (must be 44100 Hz)
    """
    
    request_id: str
    channels: int
    num_samples: int
    dtype: np.float32
    waveform: NDArray[np.float32]
    sample_rate: int = 44100
    version: Literal["output_audio_file_v1"] = "output_audio_file_v1"

class AudioChunk(BaseModel):
    """
    Represents a chunk of audio data. The audio chunk SHALL adhere to the following specifications:
    
    Boundary: Chunking -> Transforms
    Version: audio_chunk_v1
    
    Constraints:
        - File must be a valid WAV format
        - Sample rate must be 44.1kHz
        - Bit depth must be 32-bit
        - Channels must be mono
        - Sample count must be <= 1024
        - Chunk index must be non-negative
    
    Data Fields:
        - request_id (str): Unique identifier for the request
        - channels (int): Number of audio channels (must be 1 for mono)
        - num_samples (int): Number of samples in the chunk
        - chunk_index (int): Index of the chunk within the audio file
        - total_chunks (int): Total number of chunks the audio file is split into
        - dtype (np.float32): Data type of the audio samples (must be float32)
        - waveform (NDArray[np.float32]): 1D array containing the audio samples
        - sample_rate (int): Sample rate of the audio (must be 44100 Hz)
    """
    
    request_id: str
    channels: int
    num_samples: int
    chunk_index: int
    total_chunks: int
    dtype: np.float32
    waveform: NDArray[np.float32]
    sample_rate: int = 44100
    version: Literal["audio_chunk_v1"] = "audio_chunk_v1"
    
class FFTChunk(BaseModel):
    """
    Represents a chunk of audio data in the frequency domain. The frequency domain chunk SHALL adhere to the following specifications:
    
    Boundary: Transforms[FFT] -> (Tone Identifier | Channel Predictor)
    Version: fft_chunk_v1
    
    Constraints:
        - Array must have length 1024
        - dtype must be complex64
    
    Data Fields:
        - request_id (str): Unique identifier for the request
        - num_bins (int): Number of frequency bins in the FFT output (must be 1024)
        - bin_hz_resolution (float): Frequency resolution of each bin in Hz
        - dtype (np.complex64): Data type of the FFT output (must be complex64)
        - frequencies (NDArray[np.complex64]): 1D array containing the FFT output values
        - sample_rate (int): Sample rate of the original audio (must be 44100 Hz)
    """
    
    request_id: str
    num_bins: int
    bin_hz_resolution: float
    dtype: np.complex64
    frequencies: NDArray[np.complex64]
    sample_rate: int = 44100
    version: Literal["fft_chunk_v1"] = "fft_chunk_v1"

class CQTChunk(BaseModel):
    """
    Represents a chunk of audio data in the constant-Q domain. The constant-Q domain chunk SHALL adhere to the following specifications:
    
    Boundary: Transforms[CQT] -> (Tone Identifier | Channel Predictor)
    Version: cqt_chunk_v1
    
    Constraints:
        - Array must have length 1024
        - dtype must be complex64
    
    Data Fields:
        - request_id (str): Unique identifier for the request
        - num_bins (int): Number of frequency bins in the CQT output
        - bins_per_octave (int): Number of bins per octave
        - f_min (float): Minimum frequency in Hz
        - dtype (np.complex64): Data type of the CQT output (must be complex64)
        - bins (NDArray[np.complex64]): 1D array containing the CQT output values
    """
    
    request_id: str
    num_bins: int
    bins_per_octave: int
    f_min: float
    dtype: np.complex64
    bins: NDArray[np.complex64]
    version: Literal["cqt_chunk_v1"] = "cqt_chunk_v1"
    
class ChromaChunk(BaseModel):
    """
    Represents a chunk of audio data in the chroma domain. The chroma domain chunk SHALL adhere to the following specifications:
    
    Boundary: Transforms[Chroma] -> (Tone Identifier | Channel Predictor)
    Version: chroma_chunk_v1
    
    Constraints:
        - Array must have length 12
        - dtype must be float32
    
    Data Fields:
        - request_id (str): Unique identifier for the request
        - num_pitches (int): Number of pitch classes (must be 12)
        - dtype (np.float32): Data type of the chroma output (must be float32)
        - pitch_classes (NDArray[np.float32]): 1D array containing the chroma values for each pitch class
    """
    
    request_id: str
    num_pitches: int
    dtype: np.float32
    pitch_classes: NDArray[np.float32]
    version: Literal["chroma_chunk_v1"] = "chroma_chunk_v1"
    
class Spectrogram(BaseModel):
    """
    Represents a spectrogram of audio data. The spectrogram SHALL adhere to the following specifications:
    
    Boundary: Tone Identifier -> Tone Identifier
    Version: spectrogram_v1
    
    Constraints:
        - Array must have 2 dimensions
        - dtype must be float32
        - num_bins == 1024
        - num_frames == 12
    
    Data Fields:
        - request_id (str): Unique identifier for the request
        - num_bins (int): Number of frequency bins (must be 1024)
        - num_frames (int): Number of time frames (must be 12)
        - dtype (np.float32): Data type of the spectrogram (must be float32)
        - spectrogram (NDArray[np.float32]): 2D array containing the spectrogram values
        - sample_rate (int): Sample rate of the original audio (must be 44100 Hz)
    """
    
    request_id: str
    num_bins: int
    num_frames: int
    dtype: np.float32
    spectrogram: NDArray[np.float32]
    sample_rate: int = 44100
    version: Literal["spectrogram_v1"] = "spectrogram_v1"
    
class TonePrediction(BaseModel):
    """
    Represents a prediction of what tones are present in a chunk of audio data. The tone prediction SHALL adhere to the following specifications:
    
    Boundary: Tone Identifier -> Channel Predictor
    Version: tone_prediction_v1
    
    Constraints:
        - Array must have length equal to the number of SoundClassifications (currently 5)
        - dtype must be float32
    
    Data Fields:
        - request_id (str): Unique identifier for the request
        - num_classes (int): Number of sound classifications (must be 5)
        - dtype (np.float32): Data type of the tone prediction (must be float32)
        - class_probabilities (NDArray[np.float32]): 1D array containing the predicted probabilities for each sound classification
    """

    request_id: str
    num_classes: int
    dtype: np.float32
    class_probabilities: NDArray[np.float32]
    version: Literal["tone_prediction_v1"] = "tone_prediction_v1"
    
class PredictedChunk(BaseModel):
    """
    Represents a predicted chunk of audio on all classifications. The predicted chunk SHALL adhere to the following specifications:
    
    Boundary: Channel Predictor -> Channel Fuser
    Version: predicted_chunk_v1
    
    Constraints:
        - Array must be 2D with shape (num_classes, num_samples)
        - dtype must be float32
        - num_samples must be <= 1024
        - num_classes must be equal to the number of SoundClassifications (currently 5)
        
    Data Fields:
        - request_id (str): Unique identifier for the request
        - num_classes (int): Number of sound classifications (must be 5)
        - num_samples (int): Number of samples in the predicted chunk (must be <= 1024)
        - dtype (np.float32): Data type of the predicted chunk (must be float32)
        - predictions (NDArray[np.float32]): 2D array containing the predicted values for each sound classification and sample
    """
    
    request_id: str
    num_classes: int
    num_samples: int
    dtype: np.float32
    predictions: NDArray[np.float32]
    version: Literal["predicted_chunk_v1"] = "predicted_chunk_v1"
