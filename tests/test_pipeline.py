"""Comprehensive unit and integration tests for the audio processing pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pytest
import soundfile as sf

from pipeline.align import get_whisper_model, transcribe_words
from pipeline.audio_io import AudioError, load_audio
from pipeline.features import frame_tracks, summarize, word_features
from pipeline.run_pipeline import build_metrics, clean, run_pair, validate_metrics
from pipeline.schema import METRICS_SCHEMA


@pytest.fixture
def synthetic_audio_bundle(tmp_path: Path) -> dict[str, Path]:
    """Generate synthetic audio test fixtures in tmp_path (never committed to repository)."""
    sr = 16000
    duration = 3.5

    # 1. Harmonic voiced tone with pitch modulation (120 Hz to 220 Hz)
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    # Smooth frequency trajectory oscillating between 120 and 220 Hz
    f_track = 170.0 + 50.0 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2.0 * np.pi * np.cumsum(f_track) / sr
    signal = (
        0.5 * np.sin(phase)
        + 0.25 * np.sin(2.0 * phase)
        + 0.12 * np.sin(3.0 * phase)
    )

    # Apply amplitude envelope with fade-in, fade-out, and a silent gap
    envelope = np.ones_like(signal)
    fade_len = int(0.1 * sr)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)

    # Insert a 0.3s silence gap around 1.5s - 1.8s
    gap_start = int(1.5 * sr)
    gap_end = int(1.8 * sr)
    envelope[gap_start:gap_end] = 0.0

    voiced_wave = (signal * envelope * 0.7).astype(np.float32)
    tone_path = tmp_path / "synthetic_tone.wav"
    sf.write(str(tone_path), voiced_wave, sr)

    # 2. Pure silence audio (peak < 1e-4)
    silence_wave = np.zeros(int(1.5 * sr), dtype=np.float32)
    silence_path = tmp_path / "silence.wav"
    sf.write(str(silence_path), silence_wave, sr)

    # 3. Empty file (0 bytes)
    empty_path = tmp_path / "empty.wav"
    empty_path.touch()

    # 4. Corrupt file (random bytes)
    corrupt_path = tmp_path / "corrupt.wav"
    corrupt_path.write_bytes(b"NOT_A_VALID_WAV_HEADER_RANDOM_CORRUPT_BYTES_XYZ1234567890")

    # 5. Too-short audio (<0.5s duration)
    short_wave = (0.5 * np.sin(2 * np.pi * 200 * np.linspace(0, 0.2, int(0.2 * sr)))).astype(np.float32)
    short_path = tmp_path / "too_short.wav"
    sf.write(str(short_path), short_wave, sr)

    # 6. Missing path
    missing_path = tmp_path / "non_existent.wav"

    # 7. Reference transcript file
    transcript_path = tmp_path / "sample_transcript.txt"
    transcript_path.write_text("Um, hello world and welcome.", encoding="utf-8")

    return {
        "tone": tone_path,
        "silence": silence_path,
        "empty": empty_path,
        "corrupt": corrupt_path,
        "too_short": short_path,
        "missing": missing_path,
        "transcript": transcript_path,
    }


def test_1_fixture_validates_against_schema() -> None:
    """1. Ensure pipeline/fixtures/sample_metrics.json conforms to METRICS_SCHEMA and has all contract keys."""
    fixture_path = Path("pipeline/fixtures/sample_metrics.json")
    assert fixture_path.exists(), f"Fixture file not found at {fixture_path}"

    with open(fixture_path, encoding="utf-8") as f:
        data = json.load(f)

    # Validate against schema
    jsonschema.validate(instance=data, schema=METRICS_SCHEMA)

    # Assert top-level contract keys
    assert data["schema_version"] == "1.0"
    assert isinstance(data["sample_id"], str)
    assert isinstance(data["transcript"], str)
    assert isinstance(data["duration_seconds"], (int, float))
    assert isinstance(data["segments"], list)
    assert len(data["segments"]) == 5
    assert isinstance(data["summary_metrics"], dict)

    # Assert segment structure and features
    for seg in data["segments"]:
        assert "start" in seg and "end" in seg and "text" in seg and "features" in seg
        features = seg["features"]
        assert "f0_median_st" in features
        assert "f0_range_st" in features
        assert "rms_relative_db" in features
        assert "speech_rate_wpm" in features
        assert "whisper_avg_logprob" in features
        assert "pause_after_seconds" in features

    # Assert null handling demonstration (final-word pause is null)
    last_seg_pause = data["segments"][-1]["features"]["pause_after_seconds"]
    assert last_seg_pause is None

    # Assert summary_metrics keys
    summary = data["summary_metrics"]
    for key in ["words_per_minute", "filler_count", "median_pause_seconds", "long_pause_count"]:
        assert key in summary


def test_2_load_audio_succeeds_on_synthetic_tone(synthetic_audio_bundle: dict[str, Path]) -> None:
    """2. load_audio succeeds on synthetic tone: mono, 16000 Hz, float32, original file unchanged."""
    tone_path = synthetic_audio_bundle["tone"]
    initial_bytes = tone_path.read_bytes()

    y, sr = load_audio(tone_path)

    assert sr == 16000
    assert isinstance(y, np.ndarray)
    assert y.dtype == np.float32
    assert y.ndim == 1
    assert len(y) > 0

    # Ensure original file bytes are unaltered
    final_bytes = tone_path.read_bytes()
    assert initial_bytes == final_bytes


def test_3_load_audio_raises_readable_audio_error(synthetic_audio_bundle: dict[str, Path], tmp_path: Path) -> None:
    """3. load_audio raises AudioError with a readable message for invalid files."""
    # Silent file
    with pytest.raises(AudioError, match="silent"):
        load_audio(synthetic_audio_bundle["silence"])

    # Empty file
    with pytest.raises(AudioError, match="empty"):
        load_audio(synthetic_audio_bundle["empty"])

    # Corrupt file
    with pytest.raises(AudioError, match="Corrupt|unsupported"):
        load_audio(synthetic_audio_bundle["corrupt"])

    # Missing file
    with pytest.raises(AudioError, match="does not exist"):
        load_audio(synthetic_audio_bundle["missing"])

    # Too short audio
    with pytest.raises(AudioError, match="shorter than required minimum"):
        load_audio(synthetic_audio_bundle["too_short"])

    # Directory path
    with pytest.raises(AudioError, match="directory"):
        load_audio(tmp_path)


def test_4_frame_tracks_pitch_and_energy(synthetic_audio_bundle: dict[str, Path]) -> None:
    """4. frame_tracks: F0 semitone median near 0, no NaN leak after clean, unvoiced input handled gracefully."""
    y, sr = load_audio(synthetic_audio_bundle["tone"])
    tracks = frame_tracks(y, sr)

    f0_st = tracks["f0_st"]
    voiced_st = f0_st[~np.isnan(f0_st)]
    assert len(voiced_st) > 0

    # Semitone median relative to recording median should be near 0
    semitone_median = float(np.median(voiced_st))
    assert abs(semitone_median) < 0.1

    # Relative RMS should have 0 dB as upper bound (approx 95th percentile)
    rms_rel = tracks["rms_rel_db"]
    assert float(np.percentile(rms_rel, 95)) == pytest.approx(0.0, abs=0.1)

    # Clean ensures no NaN or numpy scalar leaks
    cleaned = clean(tracks)
    for val in cleaned.values():
        if isinstance(val, list):
            for item in val:
                assert item is None or isinstance(item, (int, float, str))

    # All-unvoiced (white noise) handling without crash
    noise = (np.random.default_rng(42).standard_normal(int(1.0 * sr)) * 0.05).astype(np.float32)
    noise_tracks = frame_tracks(noise, sr)
    cleaned_noise = clean(noise_tracks)
    assert cleaned_noise["median_f0_hz"] is None
    assert all(x is None for x in cleaned_noise["f0_st"])


def test_5_word_features_and_summarize_on_hand_built_words() -> None:
    """5. word_features/summarize: pause, filler_count (um/uh counted, 'like'/'so' not counted), long_pause_count, wpm."""
    duration = 4.0

    # Simulated frame tracks
    num_frames = int(duration * 100)
    frame_times = np.linspace(0, duration, num_frames)
    f0_st = np.zeros(num_frames, dtype=np.float64)  # constant 0 semitones
    rms_rel_db = np.full(num_frames, -3.0, dtype=np.float64)
    tracks = {
        "frame_times": frame_times,
        "f0_st": f0_st,
        "rms_rel_db": rms_rel_db,
        "median_f0_hz": 150.0,
    }

    # Hand-built word units:
    # 1. "Um,"  [0.2, 0.5] -> pause: 0.7 - 0.5 = 0.20s
    # 2. "like" [0.7, 1.0] -> pause: 1.1 - 1.0 = 0.10s
    # 3. "hello"[1.1, 1.5] -> pause: 2.4 - 1.5 = 0.90s (> 0.7s, long pause)
    # 4. "so"   [2.4, 2.7] -> pause: 3.0 - 2.7 = 0.30s
    # 5. "uh"   [3.0, 3.4] -> last unit, pause: null
    units = [
        {"start": 0.2, "end": 0.5, "text": "Um,", "logprob": -0.15},
        {"start": 0.7, "end": 1.0, "text": "like", "logprob": -0.10},
        {"start": 1.1, "end": 1.5, "text": "hello", "logprob": -0.05},
        {"start": 2.4, "end": 2.7, "text": "so", "logprob": -0.12},
        {"start": 3.0, "end": 3.4, "text": "uh", "logprob": -0.20},
    ]

    segments = word_features(units, tracks, duration_seconds=duration, alignment="word")

    # Check pauses
    assert segments[0]["features"]["pause_after_seconds"] == pytest.approx(0.20, abs=0.01)
    assert segments[1]["features"]["pause_after_seconds"] == pytest.approx(0.10, abs=0.01)
    assert segments[2]["features"]["pause_after_seconds"] == pytest.approx(0.90, abs=0.01)
    assert segments[3]["features"]["pause_after_seconds"] == pytest.approx(0.30, abs=0.01)
    assert segments[4]["features"]["pause_after_seconds"] is None  # last unit null

    # Check summarize
    summary = summarize(segments, transcript="Um, like hello so uh", duration_seconds=duration, alignment="word")

    # Fillers: only "Um," and "uh" should be counted (2). "like" and "so" must NOT be counted!
    assert summary["filler_count"] == 2

    # Long pause count: only the 0.90s pause (> 0.7s)
    assert summary["long_pause_count"] == 1

    # Median pause: median of [0.20, 0.10, 0.90, 0.30] -> sorted [0.10, 0.20, 0.30, 0.90] -> median is 0.25
    assert summary["median_pause_seconds"] == pytest.approx(0.25, abs=0.01)

    # Words per minute over first-start (0.2) to last-end (3.4): duration = 3.2s, 5 words -> (5 / 3.2) * 60 = 93.75
    assert summary["words_per_minute"] == pytest.approx(93.75, abs=0.1)


def test_6_clean_sanitizes_nan_inf_and_numpy_types() -> None:
    """6. clean() converts NaN, inf, and numpy types; json.dumps(allow_nan=False) succeeds."""
    messy_data = {
        "float_val": np.float32(3.14),
        "int_val": np.int64(42),
        "bool_val": np.bool_(True),
        "nan_np": np.nan,
        "nan_py": float("nan"),
        "inf_py": float("inf"),
        "neg_inf_py": float("-inf"),
        "array": np.array([1.0, np.nan, 3.0]),
        "nested": {
            "inner_nan": np.nan,
            "inner_num": np.float64(2.718),
        },
    }

    cleaned = clean(messy_data)

    assert cleaned["float_val"] == pytest.approx(3.14, abs=1e-3)
    assert cleaned["int_val"] == 42 and isinstance(cleaned["int_val"], int)
    assert cleaned["bool_val"] is True and isinstance(cleaned["bool_val"], bool)
    assert cleaned["nan_np"] is None
    assert cleaned["nan_py"] is None
    assert cleaned["inf_py"] is None
    assert cleaned["neg_inf_py"] is None
    assert cleaned["array"] == [1.0, None, 3.0]
    assert cleaned["nested"]["inner_nan"] is None
    assert cleaned["nested"]["inner_num"] == pytest.approx(2.718, abs=1e-3)

    # Must serialize strictly without NaN/Infinity
    serialized = json.dumps(cleaned, allow_nan=False)
    assert "NaN" not in serialized
    assert "Infinity" not in serialized


def test_7_build_metrics_deterministic_with_monkeypatch(
    synthetic_audio_bundle: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """7. build_metrics end-to-end on synthetic tone with monkeypatched Whisper: deterministic and valid."""
    fixed_units = [
        {"start": 0.2, "end": 0.7, "text": "hello", "logprob": -0.15},
        {"start": 0.9, "end": 1.4, "text": "world", "logprob": -0.10},
    ]

    def mock_transcribe_words(path: str | Path, model_size: str = "base.en") -> dict[str, Any]:
        return {"alignment": "word", "units": fixed_units}

    monkeypatch.setattr("pipeline.run_pipeline.transcribe_words", mock_transcribe_words)

    tone_path = synthetic_audio_bundle["tone"]
    res1 = build_metrics(tone_path, "hello world", "sample_test_01")
    res2 = build_metrics(tone_path, "hello world", "sample_test_01")

    # Assert exact byte/value determinism
    assert res1 == res2
    assert json.dumps(res1, sort_keys=True) == json.dumps(res2, sort_keys=True)

    # Validate against schema
    validate_metrics(res1)
    assert res1["alignment"] == "word"
    assert len(res1["segments"]) == 2


def test_8_cli_exit_code_2_for_missing_file(tmp_path: Path) -> None:
    """8. CLI exits with code 2 and readable stderr message for a missing audio file."""
    fake_audio = tmp_path / "missing_file_for_cli.wav"
    transcript_file = tmp_path / "dummy_script.txt"
    transcript_file.write_text("Test transcript.", encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "pipeline.run_pipeline",
        str(fake_audio),
        str(transcript_file),
        "cli_test_sample",
        "--out-dir",
        str(tmp_path / "out"),
    ]

    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)

    assert proc.returncode == 2
    assert "Audio error:" in proc.stderr
    assert "does not exist" in proc.stderr


def test_batch_run_pair_validates_transcript(synthetic_audio_bundle: dict[str, Path], tmp_path: Path) -> None:
    """Ensure run_pair refuses missing or empty transcript files."""
    tone_path = synthetic_audio_bundle["tone"]
    empty_transcript = tmp_path / "empty_transcript.txt"
    empty_transcript.write_text("   ", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        run_pair(tone_path, tone_path, empty_transcript, "b1", "p1")

    missing_transcript = tmp_path / "non_existent_transcript.txt"
    with pytest.raises(ValueError, match="not found"):
        run_pair(tone_path, tone_path, missing_transcript, "b1", "p1")


@pytest.mark.slow
def test_9_real_whisper_smoke_test(synthetic_audio_bundle: dict[str, Path]) -> None:
    """9. Optional real-Whisper smoke test; automatically skipped if model unavailable offline."""
    try:
        get_whisper_model("base.en")
    except Exception as err:  # noqa: BLE001
        pytest.skip(f"faster-whisper base.en model is not available offline: {err}")

    result = transcribe_words(synthetic_audio_bundle["tone"], model_size="base.en")
    assert "alignment" in result
    assert result["alignment"] in ("word", "phrase", "energy_phrase")
    assert isinstance(result["units"], list)
