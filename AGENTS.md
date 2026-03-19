# Agent Guidelines for wyoming-faster-whisper

## Project Overview

Wyoming protocol server for speech-to-text using multiple backends:
- **faster-whisper**: CTranslate2-based Whisper implementation
- **kyutai-stt**: Kyutai's STT model (MLX-based, Apple Silicon optimized)
- **transformer**: HuggingFace transformers Whisper implementation
- **whisper-mps**: whisper-mps for Apple Silicon MPS acceleration

## Code Style

- **Formatting**: black with 88 line length, isort with black profile
- **Type hints**: Required (`disallow_untyped_defs = true` in mypy config)
- **Python version**: 3.8.1 - 3.12

## Commands

```bash
# Run locally
script/run --model tiny-int8 --language en --uri 'tcp://0.0.0.0:10300'

# Type checking
mypy wyoming_faster_whisper/

# Testing
pytest tests/

# Linting (dev dependencies)
black --check wyoming_faster_whisper/
isort --check wyoming_faster_whisper/
flake8 wyoming_faster_whisper/
```

## Architecture

### Entry Point
- `wyoming_faster_whisper/__main__.py`: CLI entry, model selection based on `--model-type`

### Event Handlers
Each model type has its own handler in `wyoming_faster_whisper/`:
- `handler.py` - FasterWhisperEventHandler (faster-whisper backend)
- `kyutai_stt_handler.py` - KyutaiSttEventHandler + KyutaiSttModel
- `whisper_mps_event_handler.py` - WhisperMpsEventHandler
- `transformers_whisper.py` - TransformersWhisperEventHandler + TransformersWhisperModel

### Wyoming Protocol
- Uses `wyoming.audio` for AudioChunk/AudioStop events
- Uses `wyoming.asr` for Transcribe/Transcript events
- Uses `wyoming.info` for Describe/Info events

### Key Patterns
- All handlers inherit from `AsyncEventHandler`
- Audio chunks are accumulated in memory, transcription occurs on AudioStop
- Model inference uses `asyncio.Lock` to prevent concurrent inference
- Language can be set per-request via Transcribe event

## Dependencies

Core:
- `faster-whisper>=1.1.0,<2`
- `wyoming>=1.5.3`
- `moshi_mlx>=0.3.0`
- `numpy>=2.2.6`
- `scipy>=1.16.2`
- `whisper-mps` (git dependency)

Optional:
- `transformers[torch]` for transformer backend
