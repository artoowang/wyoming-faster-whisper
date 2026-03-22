---
name: test-glm-asr-wyoming-server
description: "Test GLM ASR Wyoming Server"
version: 1.0.0
author: "Chunpo Wang"
---

# Test GLM ASR Wyoming Server

Test the GLM ASR Wyoming STT server by running it and sending audio files for transcription.

## When to Use

Use this skill when you need to:
- Test the GLM ASR backend of the Wyoming STT server
- Verify the server starts correctly
- Check transcription output from the GLM ASR model

## Prerequisites

Use `initial_prompt.md` for the default initial prompt, unless the user
specifies something else.

## Workflow

1. Start the Wyoming server on a port (avoid 10300 as it's used by system daemon):
```bash
source .venv/bin/activate && script/run --model "zai-org/GLM-ASR-Nano-2512" --model-type glm-asr --uri "tcp://0.0.0.0:10301" --log-format "%(asctime)s [%(levelname)s] %(name)s: %(message)s" --debug --initial-prompt <prompt> >/tmp/log 2>&1 &
```

1. Look at the log every second, until the log says "Ready".

1. In another terminal, send an audio file to test:
```bash
source .venv/bin/activate && python -m tests.test_wyoming --ip localhost --port 10301 --audio_file <path_to_wav>
```

1. After the test, remember to kill the server:
```bash
pkill -u "$USER" -f wyoming_faster_whisper
```

## Available Audio Files

Example audio files for testing:
- `tests/clips/turn_off_tv.wav`

## Troubleshooting

- If the server fails to start, check if the port is already in use
