---
name: test-speech-against-groundtruth
description: "Test Speech Against Ground Truth"
version: 1.0.0
author: "Chunpo Wang"
---

# Test Speech Against Ground Truth

Test transcription accuracy by running audio files through a running Wyoming STT server and comparing against known ground truths.

## When to Use

Use this skill when you need to:
- Evaluate STT model accuracy
- Benchmark different STT backends
- Verify transcription quality after changes

## Prerequisites

See AGENTS.md for starting a Wyoming STT server. Run it on port 10301.

Backends that can be used to test:
- `whisper-mps`

## Test a Single Audio File

```bash
python -m tests.test_wyoming --ip localhost --port 10301 --audio_file <path_to_wav>
```

## Batch Test Against Ground Truths

```bash
python -m tests.batch_test_ground_truths --ip localhost --port 10301
```

## Ground Truths

See `tests/ground_truths.md` for the list of audio files and their expected transcriptions.

Audio files are located at: `/Users/ollama/log/wyoming-stt-audio-debug/`

## Pass/Fail Criteria

A transcription is considered a **PASS** if:
- Only minor punctuation differences (case, periods, question marks)
- Acceptable numeric conversion (e.g., "one" → "1")
- Empty audio (marked `<empty>` in ground truth) produces hallucination

A transcription is considered a **FAIL** if:
- Semantic errors: words changed to different meaning
- Opposite meaning: "on" → "off" or similar
- Hallucination on non-empty audio (e.g., "and so on")
- Empty transcription on non-empty audio
- Completely wrong transcription

## Testing Tips

- Use `--debug` flag on server for verbose logging
- Test one backend at a time for fair comparison
- Use `--target_rate` if the audio file sample rate differs from what the server expects
