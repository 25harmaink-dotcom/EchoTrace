"""JSON Schema for metrics.json contract."""

METRICS_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "SpeechMetrics",
    "type": "object",
    "required": [
        "schema_version",
        "sample_id",
        "transcript",
        "duration_seconds",
        "segments",
        "summary_metrics",
    ],
    "properties": {
        "schema_version": {"type": "string"},
        "sample_id": {"type": "string"},
        "transcript": {"type": "string"},
        "duration_seconds": {"type": "number"},
        "alignment": {
            "type": "string",
            "enum": ["word", "phrase", "energy_phrase"],
        },
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["start", "end", "text", "features"],
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "text": {"type": "string"},
                    "features": {
                        "type": "object",
                        "required": [
                            "f0_median_st",
                            "f0_range_st",
                            "rms_relative_db",
                            "speech_rate_wpm",
                            "whisper_avg_logprob",
                            "pause_after_seconds",
                        ],
                        "properties": {
                            "f0_median_st": {"type": ["number", "null"]},
                            "f0_range_st": {"type": ["number", "null"]},
                            "rms_relative_db": {"type": ["number", "null"]},
                            "speech_rate_wpm": {"type": ["number", "null"]},
                            "whisper_avg_logprob": {"type": ["number", "null"]},
                            "pause_after_seconds": {"type": ["number", "null"]},
                        },
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
            },
        },
        "summary_metrics": {
            "type": "object",
            "required": [
                "words_per_minute",
                "filler_count",
                "median_pause_seconds",
                "long_pause_count",
            ],
            "properties": {
                "words_per_minute": {"type": ["number", "null"]},
                "filler_count": {"type": "integer"},
                "median_pause_seconds": {"type": ["number", "null"]},
                "long_pause_count": {"type": "integer"},
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}
