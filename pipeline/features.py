"""Acoustic and prosodic feature extraction module."""

from __future__ import annotations

import logging
import string
from typing import Any

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# Strict set of filler tokens (excluding "like" and "so" to avoid false positives)
FILLER_TOKENS: frozenset[str] = frozenset(
    {"um", "uh", "umm", "uhh", "er", "erm", "ah", "hmm"}
)


def frame_tracks(y: np.ndarray, sr: int = 16000) -> dict[str, Any]:
    """Compute frame-level pitch (F0 in semitones vs median) and energy (RMS relative dB).

    Uses librosa.pyin for pitch tracking with:
      - fmin = 65 Hz
      - fmax = 500 Hz
      - hop_length = 160 samples (10 ms at 16 kHz)
      - frame_length = 1024 samples (64 ms at 16 kHz)

    F0 is normalized to semitones relative to the recording's nanmedian F0:
      st = 12 * log2(f0 / median_f0)
    All-unvoiced recordings return null/NaN tracks without crashing.

    RMS frame energy is computed in dB relative to the 95th-percentile frame level:
      rms_rel_db = 20 * log10(rms) - p95_db

    Parameters
    ----------
    y : np.ndarray
        Mono 16 kHz audio samples as float32.
    sr : int
        Audio sampling rate in Hz (default: 16000).

    Returns
    -------
    dict[str, Any]
        Dictionary with:
          - "frame_times": np.ndarray of shape (N,) with timestamps in seconds
          - "f0_st": np.ndarray of shape (N,) with semitones relative to median (or NaNs)
          - "rms_rel_db": np.ndarray of shape (N,) with relative dB levels
          - "median_f0_hz": float | None, median F0 in Hz
    """
    hop_length = 160
    frame_length = 1024

    # 1. Pitch tracking with pyin
    try:
        f0, _, _ = librosa.pyin(
            y,
            fmin=65,
            fmax=500,
            sr=sr,
            hop_length=hop_length,
            frame_length=frame_length,
        )
    except Exception as err:  # noqa: BLE001
        logger.warning("Pitch extraction with pyin failed (%s); treating as unvoiced.", err)
        num_frames = 1 + int(len(y) // hop_length)
        f0 = np.full(num_frames, np.nan, dtype=np.float64)

    # Calculate frame timestamps in seconds
    frame_times = librosa.frames_to_time(
        np.arange(len(f0)), sr=sr, hop_length=hop_length
    )

    # Normalize F0 to semitones relative to recording nanmedian
    valid_f0 = f0[~np.isnan(f0) & (f0 > 0)]
    f0_st = np.full_like(f0, np.nan, dtype=np.float64)
    median_f0_hz: float | None = None

    if len(valid_f0) > 0:
        med = float(np.nanmedian(valid_f0))
        if med > 0:
            median_f0_hz = med
            mask = ~np.isnan(f0) & (f0 > 0)
            # Guard zero or negative frequencies
            f0_st[mask] = 12.0 * np.log2(f0[mask] / med)

    # 2. RMS energy tracking
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Ensure lengths match exactly
    min_len = min(len(f0), len(rms), len(frame_times))
    f0_st = f0_st[:min_len]
    rms = rms[:min_len]
    frame_times = frame_times[:min_len]

    # Clamp RMS to avoid log of zero, compute dB and relative to 95th percentile
    rms_clamped = np.maximum(rms, 1e-9)
    rms_db = 20.0 * np.log10(rms_clamped)
    p95_db = float(np.percentile(rms_db, 95))
    rms_rel_db = rms_db - p95_db

    return {
        "frame_times": frame_times,
        "f0_st": f0_st,
        "rms_rel_db": rms_rel_db,
        "median_f0_hz": median_f0_hz,
    }


def word_features(
    units: list[dict[str, Any]],
    tracks: dict[str, Any],
    duration_seconds: float,
    alignment: str = "word",
) -> list[dict[str, Any]]:
    """Compute per-unit features matching the shared contract.

    Features computed per segment:
      - f0_median_st: float | null (semitones vs recording median F0)
      - f0_range_st: float | null (p95 - p5 of semitone track; requires >= 5 voiced frames else null)
      - rms_relative_db: float | null (dB relative to 95th-percentile frame level)
      - speech_rate_wpm: float | null (local words/min in 3s centered window; null if alignment != 'word')
      - whisper_avg_logprob: float | null (from transcription or null)
      - pause_after_seconds: float | null (next.start - this.end clipped >= 0, null for last unit)

    Parameters
    ----------
    units : list[dict[str, Any]]
        List of aligned units with "start", "end", "text", "logprob".
    tracks : dict[str, Any]
        Frame tracks from frame_tracks() with "frame_times", "f0_st", "rms_rel_db".
    duration_seconds : float
        Total duration of the audio in seconds.
    alignment : str
        Alignment mode: "word", "phrase", or "energy_phrase".

    Returns
    -------
    list[dict[str, Any]]
        List of segment dictionaries with "start", "end", "text", and "features".
    """
    frame_times: np.ndarray = tracks["frame_times"]
    f0_st: np.ndarray = tracks["f0_st"]
    rms_rel_db: np.ndarray = tracks["rms_rel_db"]

    segments: list[dict[str, Any]] = []
    num_units = len(units)

    for i, unit in enumerate(units):
        start = float(unit["start"])
        end = float(unit["end"])
        text = str(unit["text"])
        center_t = (start + end) / 2.0

        # Frame mask for frames falling within this unit
        mask = (frame_times >= start) & (frame_times <= end)
        if not np.any(mask) and len(frame_times) > 0:
            # If no frame falls strictly within the interval, pick the single nearest frame
            nearest_idx = int(np.argmin(np.abs(frame_times - center_t)))
            mask = np.zeros(len(frame_times), dtype=bool)
            mask[nearest_idx] = True

        # 1. Pitch features
        seg_f0_st = f0_st[mask]
        voiced_st = seg_f0_st[~np.isnan(seg_f0_st)]

        if len(voiced_st) > 0:
            f0_median_st: float | None = float(np.median(voiced_st))
        else:
            f0_median_st = None

        if len(voiced_st) >= 5:
            p95 = float(np.percentile(voiced_st, 95))
            p5 = float(np.percentile(voiced_st, 5))
            f0_range_st: float | None = float(p95 - p5)
        else:
            f0_range_st = None

        # 2. RMS relative dB
        seg_rms = rms_rel_db[mask]
        if len(seg_rms) > 0:
            rms_relative_db: float | None = float(np.mean(seg_rms))
        else:
            rms_relative_db = None

        # 3. Speech rate in local 3-second centered window
        if alignment == "word":
            win_start = center_t - 1.5
            win_end = center_t + 1.5
            eff_start = max(0.0, win_start)
            eff_end = min(duration_seconds, win_end)
            eff_duration = eff_end - eff_start
            if eff_duration > 0:
                words_in_window = sum(
                    1 for u in units if eff_start <= u["start"] <= eff_end
                )
                speech_rate_wpm: float | None = float((words_in_window / eff_duration) * 60.0)
            else:
                speech_rate_wpm = None
        else:
            speech_rate_wpm = None

        # 4. Whisper avg logprob
        logprob = unit.get("logprob")
        whisper_avg_logprob: float | None = float(logprob) if logprob is not None else None

        # 5. Pause after seconds
        if i < num_units - 1:
            next_start = float(units[i + 1]["start"])
            pause = max(0.0, next_start - end)
            pause_after_seconds: float | None = float(round(pause, 2))
        else:
            pause_after_seconds = None

        segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "features": {
                    "f0_median_st": f0_median_st,
                    "f0_range_st": f0_range_st,
                    "rms_relative_db": rms_relative_db,
                    "speech_rate_wpm": speech_rate_wpm,
                    "whisper_avg_logprob": whisper_avg_logprob,
                    "pause_after_seconds": pause_after_seconds,
                },
            }
        )

    return segments


def summarize(
    segments: list[dict[str, Any]],
    transcript: str,
    duration_seconds: float,
    alignment: str = "word",
) -> dict[str, Any]:
    """Compute summary metrics over the entire recording.

    Metrics:
      - words_per_minute: float | null (computed over first-start to last-end; null if duration <= 0)
      - filler_count: int (counts only um, uh, umm, uhh, er, erm, ah, hmm; no like or so)
      - median_pause_seconds: float | null (median of non-null pause_after_seconds)
      - long_pause_count: int (count of pauses > 0.7 s)

    Parameters
    ----------
    segments : list[dict[str, Any]]
        List of segment dictionaries with features.
    transcript : str
        Full reference transcript.
    duration_seconds : float
        Total duration of the audio recording in seconds.
    alignment : str
        Alignment mode: "word", "phrase", or "energy_phrase".

    Returns
    -------
    dict[str, Any]
        Dictionary matching summary_metrics contract.
    """
    # 1. Words per minute over active speech duration
    if not segments:
        words_per_minute: float | None = None
    else:
        first_start = float(segments[0]["start"])
        last_end = float(segments[-1]["end"])
        speech_duration = last_end - first_start

        if speech_duration <= 0:
            words_per_minute = None
        else:
            if alignment == "word":
                word_count = len(segments)
            else:
                # Phrase or energy_phrase: count words from transcript if available
                words = [w for w in transcript.split() if w.strip()]
                word_count = len(words) if words else len(segments)

            words_per_minute = float((word_count / speech_duration) * 60.0)

    # 2. Filler count
    # Count strictly allowed filler tokens, stripping punctuation and lowercasing
    filler_count = 0
    punctuation_strip = str.maketrans("", "", string.punctuation)

    # If segments contain actual speech words, count from segment texts
    text_source = [s["text"] for s in segments if s["text"] != "[speech]"]
    if not text_source and transcript.strip():
        text_source = transcript.split()

    for text in text_source:
        tokens = text.split()
        for tok in tokens:
            cleaned = tok.lower().translate(punctuation_strip)
            if cleaned in FILLER_TOKENS:
                filler_count += 1

    # 3. Pauses
    pauses = [
        float(s["features"]["pause_after_seconds"])
        for s in segments
        if s["features"].get("pause_after_seconds") is not None
    ]

    if pauses:
        median_pause_seconds: float | None = float(np.median(pauses))
        long_pause_count = int(sum(1 for p in pauses if p > 0.7))
    else:
        median_pause_seconds = None
        long_pause_count = 0

    return {
        "words_per_minute": words_per_minute,
        "filler_count": filler_count,
        "median_pause_seconds": median_pause_seconds,
        "long_pause_count": long_pause_count,
    }
