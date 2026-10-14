"""Audio processing, alignment, and acoustic feature extraction pipeline."""

from pipeline.align import transcribe_words
from pipeline.audio_io import AudioError, load_audio
from pipeline.features import frame_tracks, summarize, word_features
from pipeline.run_pipeline import build_metrics, clean, run_pair, validate_metrics
from pipeline.schema import METRICS_SCHEMA

__all__ = [
    "METRICS_SCHEMA",
    "AudioError",
    "build_metrics",
    "clean",
    "frame_tracks",
    "load_audio",
    "run_pair",
    "summarize",
    "transcribe_words",
    "validate_metrics",
    "word_features",
]
