"""Audio loading and validation module."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE: int = 16000
MIN_DURATION_SECONDS: float = 0.5
SILENCE_PEAK_THRESHOLD: float = 1e-4
CLIPPING_THRESHOLD: float = 0.99
CLIPPING_PERCENT_LIMIT: float = 0.01  # >1% of samples


class AudioError(Exception):
    """Raised when an audio file cannot be loaded, decoded, or validated."""


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file, convert it to 16 kHz mono float32, and validate it.

    Parameters
    ----------
    path : str | Path
        Path to the target audio file.

    Returns
    -------
    tuple[np.ndarray, int]
        y : np.ndarray
            Mono audio waveform as 1D float32 array, normalized in [-1.0, 1.0].
        sr : int
            Sampling rate in Hz (always 16000).

    Raises
    ------
    AudioError
        If the file is missing, is a directory, cannot be decoded/is corrupt,
        contains zero samples, has duration < 0.5 seconds, or is silent (peak < 1e-4).
    """
    audio_path = Path(path)

    # 1. Missing file check
    if not audio_path.exists():
        raise AudioError(f"Audio file does not exist: {audio_path}")

    # 2. Directory check
    if audio_path.is_dir():
        raise AudioError(f"Specified path is a directory, not an audio file: {audio_path}")

    # 3. Empty file (0 bytes) check
    try:
        if audio_path.stat().st_size == 0:
            raise AudioError(f"Audio file is empty (0 bytes): {audio_path}")
    except OSError as err:
        raise AudioError(f"Failed to inspect audio file '{audio_path}': {err}") from err

    # 4. Decode audio using soundfile (read-only)
    try:
        data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    except Exception as err:
        raise AudioError(f"Corrupt or unsupported audio file '{audio_path}': {err}") from err

    # 5. Zero-length check
    if data is None or len(data) == 0:
        raise AudioError(f"Audio file contains zero audio samples: {audio_path}")

    # Convert multi-channel to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample to 16 kHz if necessary
    if sample_rate != TARGET_SAMPLE_RATE:
        data = librosa.resample(data, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
        sample_rate = TARGET_SAMPLE_RATE

    data = np.ascontiguousarray(data, dtype=np.float32)

    # 6. Audio length check (minimum 0.5 s)
    duration_s = float(len(data)) / float(sample_rate)
    if duration_s < MIN_DURATION_SECONDS:
        raise AudioError(
            f"Audio duration ({duration_s:.3f} s) is shorter than required minimum of {MIN_DURATION_SECONDS} s: {audio_path}"
        )

    # 7. Silent audio check (peak < 1e-4)
    peak = float(np.max(np.abs(data)))
    if peak < SILENCE_PEAK_THRESHOLD:
        raise AudioError(
            f"Audio is completely silent (peak amplitude {peak:.2e} < {SILENCE_PEAK_THRESHOLD}): {audio_path}"
        )

    # Heavy clipping warning (>1% samples at |x| >= 0.99)
    clipping_ratio = float(np.mean(np.abs(data) >= CLIPPING_THRESHOLD))
    if clipping_ratio > CLIPPING_PERCENT_LIMIT:
        msg = (
            f"Heavy clipping detected in '{audio_path}': "
            f"{clipping_ratio * 100:.2f}% of samples exceed amplitude {CLIPPING_THRESHOLD}."
        )
        logger.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    return data, sample_rate
