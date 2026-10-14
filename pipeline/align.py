"""Transcription and alignment module using faster-whisper with deterministic fallback."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import librosa

from pipeline.audio_io import load_audio

logger = logging.getLogger(__name__)

# Module-level model cache to avoid re-initializing Whisper on repeated calls
_MODEL_CACHE: dict[str, Any] = {}


def get_whisper_model(model_size: str = "base.en") -> Any:
    """Retrieve or initialize a cached faster-whisper WhisperModel instance.

    Parameters
    ----------
    model_size : str
        Whisper model identifier (default: "base.en").

    Returns
    -------
    WhisperModel
        Cached faster-whisper model instance.
    """
    if model_size not in _MODEL_CACHE:
        from faster_whisper import WhisperModel

        logger.info("Initializing faster-whisper model '%s' on CPU with int8 quantization.", model_size)
        _MODEL_CACHE[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _MODEL_CACHE[model_size]


def transcribe_words(path: str | Path, model_size: str = "base.en") -> dict[str, Any]:
    """Transcribe audio and extract word-level or phrase-level time-aligned units.

    Uses faster-whisper with deterministic parameters (beam_size=5, temperature=0.0).
    Provides progressive fallbacks:
      1. Word-level timestamps (alignment='word')
      2. Whisper segment-level phrase timestamps (alignment='phrase')
      3. Energy-based silence segmentation (alignment='energy_phrase') if Whisper is offline/unavailable.

    Parameters
    ----------
    path : str | Path
        Path to the target audio file.
    model_size : str
        Model size for faster-whisper (default: "base.en").

    Returns
    -------
    dict[str, Any]
        Dictionary with:
          - "alignment": "word" | "phrase" | "energy_phrase"
          - "units": list of dicts with {"start": float, "end": float, "text": str, "logprob": float | None}
    """
    audio_path = Path(path)

    try:
        model = get_whisper_model(model_size)
        segments, _ = model.transcribe(
            str(audio_path),
            language="en",
            word_timestamps=True,
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            initial_prompt="Um, uh, so, like, you know,",
        )
        segment_list = list(segments)

        # Attempt word-level alignment
        word_units: list[dict[str, Any]] = []
        for segment in segment_list:
            if getattr(segment, "words", None):
                for word in segment.words:
                    text = word.word.strip()
                    if not text:
                        continue
                    start = round(float(word.start), 2)
                    end = round(float(word.end), 2)
                    if end <= start:
                        continue
                    logprob = float(segment.avg_logprob) if segment.avg_logprob is not None else None
                    word_units.append(
                        {
                            "start": start,
                            "end": end,
                            "text": text,
                            "logprob": logprob,
                        }
                    )

        if word_units:
            logger.info("Alignment succeeded at level: 'word' (%d units).", len(word_units))
            return {"alignment": "word", "units": word_units}

        # Fallback (1): Whisper segment-level timestamps
        logger.warning(
            "Word timestamps missing or empty; falling back to Whisper segment-level phrase alignment."
        )
        phrase_units: list[dict[str, Any]] = []
        for segment in segment_list:
            text = segment.text.strip()
            if not text:
                continue
            start = round(float(segment.start), 2)
            end = round(float(segment.end), 2)
            if end <= start:
                continue
            logprob = float(segment.avg_logprob) if segment.avg_logprob is not None else None
            phrase_units.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "logprob": logprob,
                }
            )

        if phrase_units:
            logger.info("Alignment succeeded at level: 'phrase' (%d units).", len(phrase_units))
            return {"alignment": "phrase", "units": phrase_units}

    except Exception as err:  # noqa: BLE001
        logger.warning(
            "Whisper transcription failed or model could not be loaded (%s). Using energy-based segmentation.",
            err,
        )

    # Fallback (2): Energy-based phrase segmentation (offline)
    y, sr = load_audio(audio_path)
    intervals = librosa.effects.split(y, top_db=30)
    raw_intervals: list[tuple[float, float]] = [
        (float(start_idx) / float(sr), float(end_idx) / float(sr))
        for start_idx, end_idx in intervals
    ]

    # Merge gaps < 0.25 s
    merged_intervals: list[list[float]] = []
    for start_s, end_s in raw_intervals:
        if not merged_intervals:
            merged_intervals.append([start_s, end_s])
        else:
            gap = start_s - merged_intervals[-1][1]
            if gap < 0.25:
                merged_intervals[-1][1] = end_s
            else:
                merged_intervals.append([start_s, end_s])

    if not merged_intervals:
        duration_s = float(len(y)) / float(sr)
        merged_intervals = [[0.0, duration_s]]

    energy_units: list[dict[str, Any]] = []
    for start_s, end_s in merged_intervals:
        start = round(float(start_s), 2)
        end = round(float(end_s), 2)
        if end <= start:
            continue
        energy_units.append(
            {
                "start": start,
                "end": end,
                "text": "[speech]",
                "logprob": None,
            }
        )

    logger.info("Alignment succeeded at level: 'energy_phrase' (%d units).", len(energy_units))
    return {"alignment": "energy_phrase", "units": energy_units}
