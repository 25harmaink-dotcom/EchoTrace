"""End-to-end execution script and CLI for speech metrics pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

from pipeline.align import transcribe_words
from pipeline.audio_io import AudioError, load_audio
from pipeline.features import frame_tracks, summarize, word_features
from pipeline.schema import METRICS_SCHEMA

logger = logging.getLogger(__name__)


def clean(obj: Any) -> Any:
    """Recursively clean data structures, converting NaN/inf and NumPy types to None and Python primitives.

    Guarantees strict JSON compliance with allow_nan=False.

    Parameters
    ----------
    obj : Any
        Object to clean (dict, list, tuple, primitive, or numpy type).

    Returns
    -------
    Any
        Sanitized object consisting purely of Python types and None.
    """
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return obj


def validate_metrics(d: dict[str, Any]) -> None:
    """Validate a metrics dictionary against the contract JSON schema.

    Parameters
    ----------
    d : dict[str, Any]
        Metrics dictionary to validate.

    Raises
    ------
    jsonschema.ValidationError
        If the metrics dictionary violates the contract schema.
    """
    jsonschema.validate(instance=d, schema=METRICS_SCHEMA)


def build_metrics(
    audio_path: str | Path,
    transcript: str,
    sample_id: str,
    model_size: str = "base.en",
) -> dict[str, Any]:
    """Execute complete end-to-end pipeline: load audio, align, extract features, and validate.

    Parameters
    ----------
    audio_path : str | Path
        Path to input audio file.
    transcript : str
        Reference transcript text.
    sample_id : str
        Unique identifier for this sample.
    model_size : str
        Whisper model size identifier (default: "base.en").

    Returns
    -------
    dict[str, Any]
        Contract-compliant, schema-validated metrics dictionary.
    """
    # 1. Load and validate audio
    y, sr = load_audio(audio_path)
    duration_seconds = round(float(len(y)) / float(sr), 2)

    # 2. Transcribe and align units
    align_result = transcribe_words(audio_path, model_size=model_size)
    alignment = align_result["alignment"]
    units = align_result["units"]

    # 3. Frame tracks (F0 normalized to semitones, RMS relative dB)
    tracks = frame_tracks(y, sr=sr)

    # 4. Word/segment features
    segments = word_features(
        units=units,
        tracks=tracks,
        duration_seconds=duration_seconds,
        alignment=alignment,
    )

    # 5. Summary metrics
    summary = summarize(
        segments=segments,
        transcript=transcript,
        duration_seconds=duration_seconds,
        alignment=alignment,
    )

    # 6. Assemble metrics payload
    raw_metrics: dict[str, Any] = {
        "schema_version": "1.0",
        "sample_id": sample_id,
        "transcript": transcript,
        "duration_seconds": duration_seconds,
        "alignment": alignment,
        "segments": segments,
        "summary_metrics": summary,
    }

    # 7. Clean all NaN/inf/NumPy types
    cleaned_metrics = clean(raw_metrics)

    # 8. Validate against contract schema
    validate_metrics(cleaned_metrics)

    return cleaned_metrics


def run_pair(
    baseline_audio: str | Path,
    participant_audio: str | Path,
    transcript_file: str | Path,
    baseline_id: str,
    participant_id: str,
    model_size: str = "base.en",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Batch helper to process paired baseline and participant recordings sharing the same script.

    Parameters
    ----------
    baseline_audio : str | Path
        Path to baseline audio.
    participant_audio : str | Path
        Path to participant audio.
    transcript_file : str | Path
        Path to reference transcript file.
    baseline_id : str
        Sample ID for baseline.
    participant_id : str
        Sample ID for participant.
    model_size : str
        Whisper model size identifier.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        (baseline_metrics, participant_metrics)

    Raises
    ------
    ValueError
        If transcript file is missing or empty.
    """
    t_path = Path(transcript_file)
    if not t_path.exists():
        raise ValueError(f"Transcript file not found: {transcript_file}")
    if t_path.is_dir():
        raise ValueError(f"Transcript path is a directory, not a file: {transcript_file}")

    transcript_content = t_path.read_text(encoding="utf-8").strip()
    if not transcript_content:
        raise ValueError(
            f"Transcript file is empty: {transcript_file}. Baseline and participant must share non-empty text."
        )

    baseline_metrics = build_metrics(
        audio_path=baseline_audio,
        transcript=transcript_content,
        sample_id=baseline_id,
        model_size=model_size,
    )
    participant_metrics = build_metrics(
        audio_path=participant_audio,
        transcript=transcript_content,
        sample_id=participant_id,
        model_size=model_size,
    )

    return baseline_metrics, participant_metrics


def main() -> None:
    """CLI entrypoint for running the pipeline."""
    parser = argparse.ArgumentParser(
        description="Extract acoustic features and word alignments into metrics.json."
    )
    parser.add_argument("audio", type=Path, help="Path to input audio file")
    parser.add_argument("transcript_file", type=Path, help="Path to reference transcript text file")
    parser.add_argument("sample_id", type=str, help="Unique identifier for this sample")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to write output JSON (default: outputs)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base.en",
        help="faster-whisper model size (default: base.en)",
    )

    args = parser.parse_args()

    # Read transcript file
    try:
        if not args.transcript_file.exists():
            sys.stderr.write(f"Error: Transcript file does not exist: {args.transcript_file}\n")
            sys.exit(1)
        if args.transcript_file.is_dir():
            sys.stderr.write(f"Error: Transcript path is a directory: {args.transcript_file}\n")
            sys.exit(1)
        transcript = args.transcript_file.read_text(encoding="utf-8").strip()
    except Exception as err:  # noqa: BLE001
        sys.stderr.write(f"Error: Failed to read transcript file '{args.transcript_file}': {err}\n")
        sys.exit(1)

    # Execute pipeline
    try:
        metrics = build_metrics(
            audio_path=args.audio,
            transcript=transcript,
            sample_id=args.sample_id,
            model_size=args.model,
        )
    except AudioError as err:
        sys.stderr.write(f"Audio error: {err}\n")
        sys.exit(2)
    except Exception as err:  # noqa: BLE001
        sys.stderr.write(f"Unexpected error: {err}\n")
        sys.exit(1)

    # Write output JSON with allow_nan=False
    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / f"{args.sample_id}_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, allow_nan=False, indent=2, sort_keys=False)
        print(f"Successfully generated metrics: {out_path}")
    except Exception as err:  # noqa: BLE001
        sys.stderr.write(f"Error: Failed to write output JSON to '{args.out_dir}': {err}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
