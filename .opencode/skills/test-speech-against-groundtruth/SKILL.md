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

Start the backend. Following are the backends that can be used to test.
Ask user which one to use if not specified.

Use `initial_prompt.md` for the default initial prompt, unless the user
specifies something else.

### `whisper-mps`

```bash
source .venv/bin/activate && script/run --model large --model-type whisper-mps --uri "tcp://0.0.0.0:10301" --log-format "%(asctime)s [%(levelname)s] %(name)s: %(message)s" --debug --initial-prompt <prompt> >/tmp/log 2>&1 &
```

### `glm-asr`

```bash
source .venv/bin/activate && script/run --model "zai-org/GLM-ASR-Nano-2512" --model-type glm-asr --uri "tcp://0.0.0.0:10301" --log-format "%(asctime)s [%(levelname)s] %(name)s: %(message)s" --debug --initial-prompt <prompt> >/tmp/log 2>&1 &
```

## Batch Test Against Ground Truths

```bash
python -m tests.batch_test_ground_truths --ip localhost --port 10301
```

## After the test, remember to kill the server
```bash
pkill -u "$USER" -f wyoming_faster_whisper
```

## Ground Truths

See `tests/ground_truths.md` for the list of audio files and their expected transcriptions.

Audio files are located at: `/Users/ollama/log/wyoming-stt-audio-debug/`

## Pass/Fail Criteria

A transcription is considered a **PASS** if:
- Only minor punctuation differences (case, periods, question marks)
- Non-empty audio transcribed correctly

A transcription is considered a **FAIL** if:
- Semantic errors: words changed to different meaning
- Opposite meaning: "on" → "off" or similar
- Hallucination on non-empty audio (e.g., "and so on")
- Empty transcription on non-empty audio
- Completely wrong transcription
- Numeric conversion: e.g., "9:14am" → "nine o'clock" (transcription should preserve exact values)
- Any transcription on empty audio (hallucination, greeting, etc.)

## Testing Tips

- Test one backend at a time for fair comparison
