# Audio Pipeline Documentation & Contract Notes

## 1. Purpose
The audio pipeline is responsible for:
- Robust audio input validation and preprocessing (converting to 16 kHz mono float32, screening out silence, corrupt files, zero-length files, and audio < 0.5 s).
- Speech-to-text alignment extracting time-aligned word-level units using `faster-whisper` with deterministic decoding, backed by progressive phrase-level and energy-based silence segmentation fallbacks.
- Prosodic and acoustic feature extraction per segment (fundamental frequency $F_0$ normalized to semitones relative to recording median, RMS energy in dB relative to 95th percentile, local speech rate in WPM, Whisper average log probabilities, and pause durations).
- Recording-level summary metrics (overall words per minute, filler word counts, median pause duration, and long pause count).
- Schema-validated, strict JSON serialization producing `metrics.json`.

---

## 2. How to Run

### Linux / macOS
```bash
# Run pipeline on a single audio file with its transcript:
.venv/bin/python -m pipeline.run_pipeline path/to/audio.wav path/to/transcript.txt sample_id --out-dir outputs

# Run test suite:
.venv/bin/python -m pytest -v tests/test_pipeline.py
```

### Windows (PowerShell)
```powershell
# Run pipeline on a single audio file with its transcript:
.\.venv\Scripts\python.exe -m pipeline.run_pipeline path\to\audio.wav path\to\transcript.txt sample_id --out-dir outputs

# Run test suite:
.\.venv\Scripts\python.exe -m pytest -v tests/test_pipeline.py
```

---

## 3. Input & Output Files

- **Input Audio**: Any standard audio format supported by `soundfile` (WAV, FLAC, OGG, etc.), mono or multi-channel, any sample rate. Converted in-memory to 16 kHz mono float32 without modifying the source file.
- **Input Transcript**: UTF-8 encoded plain text file matching the speech prompt.
- **Output Metrics**: `outputs/<sample_id>_metrics.json` strictly compliant with `METRICS_SCHEMA` (`schema_version`: `"1.0"`).

---

## 4. Feature Units Specification

| Field | JSON Type | Unit / Reference Scale | Description |
|---|---|---|---|
| `start` / `end` | `number` | Seconds ($s$) | Offset from audio start, rounded to 2 decimal places. |
| `duration_seconds`| `number` | Seconds ($s$) | Total recording duration. |
| `f0_median_st` | `number \| null` | Semitones ($st$) | Median pitch relative to the recording's nanmedian $F_0$: $12 \log_2(f_0 / \text{median})$. |
| `f0_range_st` | `number \| null` | Semitones ($st$) | Pitch spread ($p_{95} - p_{5}$) within segment; requires $\ge 5$ voiced frames, else `null`. |
| `rms_relative_db` | `number \| null` | Decibels ($dB$) | Mean frame RMS in dB relative to the recording's 95th percentile frame level ($dB \le 0$). |
| `speech_rate_wpm` | `number \| null` | Words per minute ($WPM$) | Words starting within a 3s window centered on the segment divided by effective duration inside recording $\times 60$. `null` if alignment is not word-level. |
| `whisper_avg_logprob` | `number \| null` | Log probability | Average log-probability reported by Whisper for the segment/word, or `null`. |
| `pause_after_seconds` | `number \| null` | Seconds ($s$) | Interval to next unit: $\max(0, \text{next.start} - \text{this.end})$. `null` for the final unit. |
| `summary_metrics.words_per_minute` | `number \| null` | Words per minute ($WPM$) | Word count divided by active duration ($\text{last.end} - \text{first.start}$) $\times 60$. `null` if duration $\le 0$. |
| `summary_metrics.filler_count` | `integer` | Count | Count of canonical filler words: `um`, `uh`, `umm`, `uhh`, `er`, `erm`, `ah`, `hmm`. |
| `summary_metrics.median_pause_seconds` | `number \| null` | Seconds ($s$) | Median duration of all non-null pauses in the recording. |
| `summary_metrics.long_pause_count` | `integer` | Count | Total count of pauses longer than 0.70 seconds. |

---

## 5. Alignment Levels & Fallback Hierarchy

The optional top-level key `"alignment"` indicates the granularity and method used:

1. **`"word"` (Primary)**:
   - Word-level timestamps extracted using `faster-whisper` (`word_timestamps=True`).
   - Units with $\text{end} \le \text{start}$ are dropped.
2. **`"phrase"` (Fallback 1)**:
   - Used if word-level timestamps are unavailable or empty.
   - Preserves Whisper segment-level boundaries and transcripts.
3. **`"energy_phrase"` (Fallback 2)**:
   - Used if the Whisper model fails to initialize or the system operates in an offline environment without cached weights.
   - Segments based on acoustic silence split (`librosa.effects.split`, `top_db=30`) with gaps $< 0.25$ s merged.
   - Text is set to `"[speech]"` and logprob to `null`. Per-word timing is never fabricated.

---

## 6. Null & Strict JSON Handling

- Unavailable, unvoiced, or indeterminate acoustic measurements are represented as JSON `null` (never `0`, never `NaN`, and never `Infinity`).
- The serialization routine uses `clean()` to recursively replace `math.isnan()`, `math.isinf()`, and NumPy numeric types with `None` and Python primitives before saving with `json.dump(..., allow_nan=False)`.

---

## 7. Determinism Settings

Whisper transcription parameters are pinned to ensure deterministic output:
- `device="cpu"`
- `compute_type="int8"`
- `beam_size=5`
- `temperature=0.0`
- `condition_on_previous_text=False`
- `vad_filter=False`
- `initial_prompt="Um, uh, so, like, you know,"` (biases Whisper to retain hesitations and filler tokens rather than filtering them).
- WhisperModel is cached at module level in `pipeline.align._MODEL_CACHE`.

---

## 8. Installed Package Versions

Versions recorded via `pip freeze` in the project virtual environment (Python 3.11.16):
- `faster-whisper==1.2.1`
- `librosa==0.11.0`
- `soundfile==0.14.0`
- `numpy==2.4.6`
- `scipy==1.17.1`
- `jsonschema==4.26.0`
- `jsonschema-specifications==2025.9.1`
- `pytest==9.1.1`
- `ruff==0.17.0`

---

## 9. Known Limitations

- **Creaky / Noisy Voices**: `librosa.pyin` is a probabilistic fundamental frequency estimator. On heavily creaky, vocal fry, or noisy audio segments, voiced frame detection may drop below the 5-frame threshold, returning `null` for `f0_range_st`.
- **Whisper Filler Filtering**: While `initial_prompt` biases Whisper toward retaining hesitation tokens, Whisper's sequence-to-sequence decoder may occasionally omit subtle or mumbled filler sounds.
- **Short Samples**: Speech rate windowing uses a centered 3-second window; for recordings under 3 seconds, the window is clamped to the recording boundaries.
- **Synthetic Test Coverage**: Integration tests run against calibrated synthetic audio and monkeypatched Whisper units for fast, offline determinism. Real audio tests should be verified against participant audio in track staging.

---

## 10. Verification Commands

To re-verify the pipeline end to end:
```bash
# 1. Bytecode compilation
.venv/bin/python -m compileall pipeline tests

# 2. Pytest suite (runs 10 unit and integration tests including smoke test)
.venv/bin/python -m pytest -v tests/test_pipeline.py

# 3. Lint and style checks
.venv/bin/python -m ruff check pipeline tests
```

---

## 11. Packages for Harsha (`requirements.txt`)

Append the following lines to the project root `requirements.txt`:
```
faster-whisper==1.2.1
jsonschema==4.26.0
librosa==0.11.0
numpy==2.4.6
pytest==9.1.1
ruff==0.17.0
scipy==1.17.1
soundfile==0.14.0
```
