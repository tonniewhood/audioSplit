
from collections import deque
from enum import Enum
from typing import Dict, Literal, Annotated

import numpy as np
from pydantic import BaseModel, model_validator, Field, field_validator
from numpydantic import NDArray, Shape
# from numpy.typing import NDArray

import common.constants as cc


# NDArrayFloat32 = Annotated[NDArray[np.float32], Field(dtype=np.float32)]

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
        - waveform (np.ndarray): 1D array containing the audio samples
        - sample_rate (int): Sample rate of the audio (must be 44100 Hz)
    """
    
    request_id: str
    filename: str
    channels: int
    num_samples: int
    num_chunks: int
    sample_rate: int
    dtype: Literal["float32"] 
    waveform: NDArray[Shape["*"], np.float32] # type: ignore[reportInvalidTypeForm]
    version: Literal["input_audio_file_v1"] = "input_audio_file_v1"

    @field_validator("waveform", mode="before")
    @classmethod
    def _cast_waveform(cls, v):
        return np.asarray(v, dtype=np.float32)
    
    async def __aiter__(self):
        """Async generator method to yield AudioChunk objects from the input audio file."""
        total_chunks = self.num_chunks
        for chunk_index in range(total_chunks):
            start_sample = chunk_index * cc.CHUNK_SIZE
            end_sample = min(start_sample + cc.CHUNK_SIZE, self.num_samples)
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
            
    # __anext__ is provided by the async generator created by __aiter__
            
    @model_validator(mode="after")
    def validate_input_audio_file(self):
        self.validate_contents()
        return self

    def validate_contents(self):
        assert self.request_id, "Request ID is required."
        assert len(self.request_id) <= cc.MAX_STR_LEN, "Request ID is too long."
        assert self.channels == 1, "Only mono audio is supported."
        assert self.num_samples > 0, "num_samples must be positive."
        assert self.num_chunks > 0, "num_chunks must be positive."
        assert self.dtype == "float32", "Audio data must be float32."
        assert isinstance(self.waveform, np.ndarray), "Audio data must be a numpy array."
        assert self.waveform.dtype == np.float32, "Audio data must be in float32 format."
        assert self.waveform.ndim == 1, "Audio data must be a 1D array. (Mono audio only)"
        assert len(self.waveform) == self.num_samples, "Mismatch between num_samples and waveform length."
        assert self.sample_rate == cc.SAMPLE_RATE, f"Sample rate must be {cc.SAMPLE_RATE} Hz."
        assert len(self.waveform) > 0, "Audio file is too short to be processed."
        expected_chunks = int(np.ceil(self.num_samples / cc.CHUNK_SIZE))
        assert self.num_chunks == expected_chunks, "num_chunks does not match num_samples."
        
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
        - waveform (np.ndarray): Array containing the audio samples
        - sample_rate (int): Sample rate of the audio (must be 44100 Hz)
    """
    
    request_id: str
    channels: int
    num_samples: int
    dtype: Literal["float32"]
    waveform: NDArray[Shape["*, *"], np.float32] # type:ignore[reportInvalidTypeForm]
    sample_rate: int = 44100
    version: Literal["output_audio_file_v1"] = "output_audio_file_v1"

    @field_validator("waveform", mode="before")
    @classmethod
    def _cast_waveform(cls, v):
        return np.asarray(v, dtype=np.float32)

    @model_validator(mode="after")
    def validate_output_audio_file(self):
        self.validate_contents()
        return self

    def validate_contents(self):
        assert self.request_id, "Request ID is required."
        assert len(self.request_id) <= cc.MAX_STR_LEN, "Request ID is too long."
        assert self.channels > 0, "channels must be positive."
        assert self.channels <= len(SoundClassifications), (
            f"channels must be <= {len(SoundClassifications)}."
        )
        assert self.sample_rate == cc.SAMPLE_RATE, f"Sample rate must be {cc.SAMPLE_RATE} Hz."
        assert self.dtype == "float32", "Audio data type must be float32."
        assert isinstance(self.waveform, np.ndarray), "Audio waveform must be a numpy array."
        assert self.waveform.dtype == np.float32, "Audio waveform must be in float32 format."
        assert self.waveform.ndim <= 8, "Audio waveform must be at most an 8D array. (one per channel)"
        assert self.waveform.ndim == self.channels, "The number of channels must match the first dimension of the waveform."
        assert self.num_samples > 0, "Output audio file is too short to be valid."
        assert self.version == "output_audio_file_v1", "Unsupported version for OutputAudioFile."

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
        - waveform (np.ndarray): 1D array containing the audio samples
        - sample_rate (int): Sample rate of the audio (must be 44100 Hz)
    """
    
    request_id: str
    channels: int
    num_samples: int
    chunk_index: int
    total_chunks: int
    sample_rate: int
    dtype: Literal["float32"]
    waveform: NDArray[Shape["1024"], np.float32] # type: ignore[reportInvalidTypeForm]
    version: Literal["audio_chunk_v1"] = "audio_chunk_v1"

    @field_validator("waveform", mode="before")
    @classmethod
    def _cast_waveform(cls, v):
        return np.asarray(v, dtype=np.float32)

    @model_validator(mode="after")
    def validate_audio_chunk(self):
        self.validate_contents()
        return self

    def validate_contents(self):
        assert self.request_id, "Request ID is required."
        assert len(self.request_id) <= cc.MAX_STR_LEN, "Request ID is too long."
        assert self.channels == 1, "Only mono audio is supported."
        assert self.num_samples == cc.CHUNK_SIZE, (
            f"Chunk size must be {cc.CHUNK_SIZE} samples."
        )
        assert self.total_chunks > 0, "Total chunks must be greater than zero."
        assert self.chunk_index >= 0, "Chunk index must be non-negative."
        assert self.chunk_index < self.total_chunks, "Chunk index must be less than total chunks."
        assert self.sample_rate == cc.SAMPLE_RATE, f"Sample rate must be {cc.SAMPLE_RATE} Hz."
        assert self.dtype == "float32", "Audio samples must be float32."
        assert isinstance(self.waveform, np.ndarray), "Waveform must be a numpy array."
        assert self.waveform.ndim == 1, "Waveform must be a 1D array."
        assert len(self.waveform) == self.num_samples, "Mismatch between num_samples and waveform length."
    
    def __anext__(self):
        pass
    
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
        - chunk_index (int): Index of the chunk within the audio file
        - total_chunks (int): Total number of chunks the audio file is split into
        - num_bins (int): Number of frequency bins in the FFT output (must be 1024)
        - bin_hz_resolution (float): Frequency resolution of each bin in Hz
        - dtype (np.complex64): Data type of the FFT output (must be complex64)
        - frequencies (np.ndarray): 1D array containing the FFT output values
        - sample_rate (int): Sample rate of the original audio (must be 44100 Hz)
    """
    
    request_id: str
    chunk_index: int
    total_chunks: int
    num_bins: int
    bin_hz_resolution: float
    dtype: Literal["complex64"]
    frequencies: NDArray[Shape[f"{cc.CHUNK_SIZE}"], np.complex64] # type: ignore[reportInvalidTypeForm]
    sample_rate: int = 44100
    version: Literal["fft_chunk_v1"] = "fft_chunk_v1"

    @field_validator("frequencies", mode="before")
    @classmethod
    def _cast_frequencies(cls, v):
        return np.asarray(v, dtype=np.complex64)

    @model_validator(mode="after")
    def validate_fft_chunk(self):
        self.validate_contents()
        return self

    def validate_contents(self):
        assert self.request_id, "Request ID is required in FFTChunk."
        assert len(self.request_id) <= cc.MAX_STR_LEN, "Request ID in FFTChunk is too long."
        assert self.chunk_index >= 0, "Chunk index in FFTChunk must be non-negative."
        assert self.total_chunks > 0, "Total chunks in FFTChunk must be greater than zero."
        assert self.chunk_index < self.total_chunks, "Chunk index in FFTChunk must be less than total chunks."
        assert self.num_bins > 0, "Number of FFT bins must be positive."
        assert self.num_bins <= cc.CHUNK_SIZE, (
            f"Number of FFT bins must not exceed {cc.CHUNK_SIZE}."
        )
        assert self.bin_hz_resolution > 0, "FFT bin frequency resolution must be positive."
        assert self.dtype == "complex64", "FFT features must be complex64."
        assert isinstance(self.frequencies, np.ndarray), "FFT features must be a numpy array."
        assert self.frequencies.ndim == 1, "FFT features must be a 1D array."
        assert self.frequencies.dtype == np.complex64, "FFT features must be in complex64 format."

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
        - chunk_index (int): Index of the chunk within the audio file
        - total_chunks (int): Total number of chunks the audio file is split into
        - num_bins (int): Number of frequency bins in the CQT output
        - bins_per_octave (int): Number of bins per octave
        - f_min (float): Minimum frequency in Hz
        - dtype (np.complex64): Data type of the CQT output (must be complex64)
        - bins (np.ndarray): 1D array containing the CQT output values
    """
    
    request_id: str
    chunk_index: int
    total_chunks: int
    num_bins: int
    bins_per_octave: int
    f_min: float
    dtype: Literal["complex64"]
    bins: NDArray[Shape[f"{cc.CHUNK_SIZE}"], np.complex64] # type: ignore[reportInvalidTypeForm]
    version: Literal["cqt_chunk_v1"] = "cqt_chunk_v1"

    @field_validator("bins", mode="before")
    @classmethod
    def _cast_bins(cls, v):
        return np.asarray(v, dtype=np.complex64)

    @model_validator(mode="after")
    def validate_cqt_chunk(self):
        self.validate_contents()
        return self

    def validate_contents(self):
        assert self.request_id, "Request ID is required in CQTChunk."
        assert len(self.request_id) <= cc.MAX_STR_LEN, "Request ID in CQTChunk is too long."
        assert self.chunk_index >= 0, "Chunk index in CQTChunk must be non-negative."
        assert self.total_chunks > 0, "Total chunks in CQTChunk must be greater than zero."
        assert self.chunk_index < self.total_chunks, "Chunk index in CQTChunk must be less than total chunks."
        assert self.num_bins > 0, "Number of CQT bins must be positive."
        assert self.num_bins <= cc.CHUNK_SIZE, (
            f"Number of CQT bins must not exceed {cc.CHUNK_SIZE}."
        )
        assert self.bins_per_octave > 0, "bins_per_octave must be positive."
        assert self.f_min > 0, "f_min must be positive."
        assert self.dtype == "complex64", "CQT features must be complex64."
        assert isinstance(self.bins, np.ndarray), "CQT features must be a numpy array."
        assert self.bins.ndim == 1, "CQT features must be a 1D array."
        assert self.bins.dtype == np.complex64, "CQT features must be in complex64 format."
    
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
        - chunk_index (int): Index of the chunk within the audio file
        - total_chunks (int): Total number of chunks the audio file is split into
        - num_pitches (int): Number of pitch classes (must be 12)
        - dtype (np.float32): Data type of the chroma output (must be float32)
        - pitch_classes (np.ndarray): 1D array containing the chroma values for each pitch class
    """
    
    request_id: str
    chunk_index: int
    total_chunks: int
    num_pitches: int
    dtype: Literal["float32"]
    pitch_classes: NDArray[Shape[f"{cc.NUM_PITCH_CLASSES}"], np.float32] # type: ignore[reportInvalidTypeForm]
    version: Literal["chroma_chunk_v1"] = "chroma_chunk_v1"

    @field_validator("pitch_classes", mode="before")
    @classmethod
    def _cast_pitch_classes(cls, v):
        return np.asarray(v, dtype=np.float32)

    @model_validator(mode="after")
    def validate_chroma_chunk(self):
        self.validate_contents()
        return self

    def validate_contents(self):
        assert self.request_id, "Request ID is required in ChromaChunk."
        assert len(self.request_id) <= cc.MAX_STR_LEN, "Request ID in ChromaChunk is too long."
        assert self.chunk_index >= 0, "Chunk index in ChromaChunk must be non-negative."
        assert self.total_chunks > 0, "Total chunks in ChromaChunk must be greater than zero."
        assert self.chunk_index < self.total_chunks, "Chunk index in ChromaChunk must be less than total chunks."
        assert self.num_pitches == cc.NUM_PITCH_CLASSES, (
            f"Chroma must have {cc.NUM_PITCH_CLASSES} pitch classes."
        )
        assert self.dtype == "float32", "Chroma features must be float32."
        assert isinstance(self.pitch_classes, np.ndarray), "Chroma features must be a numpy array."
        assert self.pitch_classes.ndim == 1, "Chroma features must be a 1D array."
        assert self.pitch_classes.dtype == np.float32, "Chroma features must be float32."
    
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
        - spectrogram (np.ndarray): 2D array containing the spectrogram values
        - sample_rate (int): Sample rate of the original audio (must be 44100 Hz)
    """
    
    request_id: str
    num_bins: int
    num_frames: int
    dtype: Literal["float32"]
    spectrogram: NDArray[Shape[f"{cc.CHUNK_SIZE}, 12"], np.float32] # type: ignore[reportInvalidTypeForm]
    sample_rate: int = 44100
    version: Literal["spectrogram_v1"] = "spectrogram_v1"

    @field_validator("spectrogram", mode="before")
    @classmethod
    def _cast_spectrogram(cls, v):
        return np.asarray(v, dtype=np.float32)

    @model_validator(mode="after")
    def validate_spectrogram(self):
        self.validate_contents()
        return self

    def validate_contents(self):
        assert self.request_id, "Request ID is required in Spectrogram."
        assert len(self.request_id) <= cc.MAX_STR_LEN, "Request ID in Spectrogram is too long."
        assert self.num_bins > 0, "num_bins must be positive."
        assert self.num_frames > 0, "num_frames must be positive."
        assert self.num_bins == cc.CHUNK_SIZE, f"num_bins must be {cc.CHUNK_SIZE}."
        assert self.num_frames == 12, "num_frames must be 12."
        assert self.dtype == "float32", "Spectrogram dtype must be float32."
        assert isinstance(self.spectrogram, np.ndarray), "Spectrogram must be a numpy array."
        assert self.spectrogram.ndim == 2, "Spectrogram must be a 2D array."
        assert self.spectrogram.dtype == np.float32, "Spectrogram array must be float32."
        assert self.spectrogram.shape == (self.num_bins, self.num_frames), (
            "Spectrogram shape must match (num_bins, num_frames)."
        )
        assert self.sample_rate == cc.SAMPLE_RATE, f"Sample rate must be {cc.SAMPLE_RATE} Hz."
    
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
        - class_probabilities (np.ndarray): 1D array containing the predicted probabilities for each sound classification
    """

    request_id: str
    num_classes: int
    dtype: Literal["float32"]
    class_probabilities: NDArray[Shape[f"{len(SoundClassifications)}, 2"], np.float32] # type:ignore[reportInvalidTypeForm]
    version: Literal["tone_prediction_v1"] = "tone_prediction_v1"

    @field_validator("class_probabilities", mode="before")
    @classmethod
    def _cast_class_probabilities(cls, v):
        return np.asarray(v, dtype=np.float32)

    @model_validator(mode="after")
    def validate_tone_prediction(self):
        self.validate_contents()
        return self
    
    def validate_contents(self):
        assert self.request_id, "Request ID is required in TonePrediction."
        assert len(self.request_id) <= cc.MAX_STR_LEN, "Request ID in TonePrediction is too long."
        assert self.num_classes == len(SoundClassifications), (f"num_classes in TonePrediction must be {len(SoundClassifications)}.")
        assert self.dtype == "float32", "Tone prediction probabilities must be float32."
        assert isinstance(self.class_probabilities, np.ndarray), "Tone prediction probabilities must be a numpy array."
        assert self.class_probabilities.ndim == 2, "Tone prediction probabilities must be a 2D array."
        assert self.class_probabilities.dtype == np.float32, "Tone prediction probabilities must be float32."
        assert self.class_probabilities.shape == (len(SoundClassifications), 2), (f"Tone prediction probabilities must have shape ({len(SoundClassifications)}, 2) for pitch and confidence.")
        assert self.version == "tone_prediction_v1", "Unsupported version for TonePrediction."
    
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
        - chunk_index (int): Index of the chunk within the audio file
        - total_chunks (int): Total number of chunks the audio file is split into
        - num_classes (int): Number of sound classifications (must be 5)
        - num_samples (int): Number of samples in the predicted chunk (must be <= 1024)
        - prediction_source (str): The source of the prediction (e.g., "fft", "cqt", "chroma")
        - dtype (np.float32): Data type of the predicted chunk (must be float32)
        - predictions (np.ndarray): 2D array containing the predicted values for each sound classification and sample
    """
    
    request_id: str
    chunk_index: int
    total_chunks: int
    num_classes: int
    num_samples: int
    prediction_source: str
    dtype: Literal["float32"]
    predictions: NDArray[Shape[f"{len(SoundClassifications)}, {cc.CHUNK_SIZE}"], np.float32] # type: ignore[reportInvalidTypeForm]
    version: Literal["predicted_chunk_v1"] = "predicted_chunk_v1"

    @field_validator("predictions", mode="before")
    @classmethod
    def _cast_predictions(cls, v):
        return np.asarray(v, dtype=np.float32)

    @model_validator(mode="after")
    def validate_predicted_chunk(self):
        self.validate_contents()
        return self

    def validate_contents(self):
        assert self.request_id, "Request ID is required in PredictedChunk."
        assert len(self.request_id) <= cc.MAX_STR_LEN, "Request ID in PredictedChunk is too long."
        assert self.chunk_index >= 0, "Chunk index in PredictedChunk must be non-negative."
        assert self.total_chunks > 0, "Total chunks in PredictedChunk must be greater than zero."
        assert self.chunk_index < self.total_chunks, "Chunk index must be less than total chunks."
        assert self.num_classes == len(SoundClassifications), (
            f"num_classes must be {len(SoundClassifications)}."
        )
        assert self.num_samples > 0, "num_samples must be positive."
        assert self.num_samples <= cc.CHUNK_SIZE, (
            f"num_samples must be <= {cc.CHUNK_SIZE}."
        )
        assert self.prediction_source in {"fft", "cqt", "chroma"}, "prediction_source must be one of 'fft', 'cqt', or 'chroma'."
        assert self.dtype == "float32", "PredictedChunk dtype must be float32."
        assert isinstance(self.predictions, np.ndarray), "Predictions must be a numpy array."
        assert self.predictions.ndim == 2, "Predictions must be a 2D array."
        assert self.predictions.dtype == np.float32, "Predictions must be float32."
        assert self.predictions.shape == (self.num_classes, self.num_samples), (
            "Predictions shape must match (num_classes, num_samples)."
        )


class FusedChunk(BaseModel):
    """
    Represents a fused prediction for a single chunk index.

    Data Fields:
        - request_id (str): Unique identifier for the request
        - chunk_index (int): Index of the chunk within the audio file
        - total_chunks (int): Total number of chunks the audio file is split into
        - num_classes (int): Number of sound classifications
        - num_samples (int): Number of samples in this fused chunk
        - dtype (np.float32): Data type of the fused chunk
        - predictions (np.ndarray): 2D array (num_classes, num_samples)
    """

    request_id: str
    chunk_index: int
    total_chunks: int
    num_classes: int
    num_samples: int
    dtype: Literal["float32"]
    predictions: NDArray[Shape[f"{len(SoundClassifications)}, {cc.CHUNK_SIZE}"], np.float32] # type: ignore[reportInvalidTypeForm]
    version: Literal["fused_chunk_v1"] = "fused_chunk_v1"

    @field_validator("predictions", mode="before")
    @classmethod
    def _cast_predictions(cls, v):
        return np.asarray(v, dtype=np.float32)

    @model_validator(mode="after")
    def validate_fused_chunk(self):
        self.validate_contents()
        return self

    def validate_contents(self):
        assert self.request_id, "Request ID is required in FusedChunk."
        assert len(self.request_id) <= cc.MAX_STR_LEN, "Request ID in FusedChunk is too long."
        assert self.chunk_index >= 0, "Chunk index in FusedChunk must be non-negative."
        assert self.total_chunks > 0, "Total chunks in FusedChunk must be greater than zero."
        assert self.chunk_index < self.total_chunks, "Chunk index must be less than total chunks."
        assert self.num_classes == len(SoundClassifications), (
            f"num_classes must be {len(SoundClassifications)}."
        )
        assert self.num_samples > 0, "num_samples must be positive."
        assert self.dtype == "float32", "FusedChunk dtype must be float32."
        assert isinstance(self.predictions, np.ndarray), "Predictions must be a numpy array."
        assert self.predictions.ndim == 2, "Predictions must be a 2D array."
        assert self.predictions.dtype == np.float32, "Predictions must be float32."
        assert self.predictions.shape == (self.num_classes, self.num_samples), (
            "Predictions shape must match (num_classes, num_samples)."
        )


class FusedAudio(BaseModel):
    """
    Tracks fused chunks for a request until all are available.

    Data Fields:
        - request_id (str): Unique identifier for the request
        - total_chunks (int): Total number of chunks expected
        - chunks (dict[int, FusedChunk]): Mapping from chunk index to fused chunk
    """

    request_id: str
    total_chunks: int
    chunks: Dict[int, FusedChunk]
    version: Literal["fused_audio_v1"] = "fused_audio_v1"

    @model_validator(mode="after")
    def validate_fused_audio(self):
        self.validate_contents()
        return self

    def validate_contents(self):
        assert self.request_id, "Request ID is required in FusedAudio."
        assert len(self.request_id) <= cc.MAX_STR_LEN, "Request ID in FusedAudio is too long."
        assert self.total_chunks > 0, "total_chunks must be positive."

# Useful classes that aren't quite dataclasses
class ChunkBuffer:
    """
    Buffer to hold incoming chunks. Has a fixed size of 8, and will drop the oldest chunk when a newer one arrives if the buffer is full.
    
    Data Fields:
        - MAX_CHUNKS (int): The maximum number of allowed chunks
        - num_chunks (int): The current number of active chunks
        - buffer (deque[np.ndarray]): The underlying chunk storage
        
    Methods: 
        - append: Adds a chunk to the buffer, evicting the oldest chunk if the buffer is full
        - get_block: Returns all chunks in the buffer as a single numpy array block
        - clear: Removes all chunks from the buffer
        - __len__: Returns the number of chunks currently stored
    """
    MAX_CHUNKS = 8

    def __init__(self, max_chunks: int | None = None):
        
        self.max_chunks = max_chunks or self.MAX_CHUNKS
        self.buffer = deque(maxlen=self.max_chunks)

    @property
    def num_chunks(self) -> int:
        return len(self.buffer)
    
    @property
    def saturated(self) -> bool:
        return self.num_chunks >= self.max_chunks

    def append(self, chunk: np.ndarray) -> None:
        self.buffer.append(chunk)

    def get_block(self) -> np.ndarray:
        if not self.buffer:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack(list(self.buffer), axis=0)

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)
