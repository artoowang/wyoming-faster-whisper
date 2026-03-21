# Plan: Add GLM-ASR Backend

## Files to Create/Modify

### 1. Create `wyoming_faster_whisper/glm_asr_event_handler.py`
- New `GlmAsrEventHandler` class (similar to existing handlers)
- Audio handling: 16kHz mono expected, accumulate chunks
- On `AudioStop`: transcribe using GLM-ASR. No streaming support needed for now.
- Language handling (GLM-ASR uses chat template, so prompt goes in messages)

### 2. Modify `wyoming_faster_whisper/__main__.py`
- Add `--model-type glm-asr` option
- Add `glm-asr` case that loads model and runs `GlmAsrEventHandler`

### 3. Test with script/run

```shell
script/run --model-type glm-asr --model "zai-org/GLM-ASR-Nano-2512" --uri "tcp://0.0.0.0:10301" --log-format "%(asctime)s [%(levelname)s] %(name)s: %(message)s" --debug --initial-prompt <prompt>
```

The prompt to use is the following:

`The following is a transcription of a user command issued to a home assistant. Common command includes 'What time is it', 'Timer, XX minutes', 'Turn on Morning Scene', 'Turn off Mos Eisley'. Common device names include 'Mos Eisley', 'Morning Scene', 'Evening Scene', 'Night Scene'`

## GLM-ASR Specific Considerations

| Aspect | Approach |
|--------|----------|
| Model loading | `AutoModel.from_pretrained(repo_id, dtype=torch.bfloat16, device_map="auto")` |
| Processor | `AutoProcessor.from_pretrained(repo_id)` |
| Audio format | 16kHz mono WAV |
| Transcription | Chat template with audio URL + system prompt |
| Device | Automatic (from `model.device`) |
| Initial prompt | Pass via chat template system message |

## Detailed Handler Structure

### Classes

1. **`GlmAsrModel`** - Wraps model/processor loading and inference
2. **`GlmAsrEventHandler`** - Handles Wyoming protocol events

### Audio Handling

```python
class GlmAsrEventHandler(AsyncEventHandler):
    def __init__(self, ...):
        # Accumulate audio chunks to temp WAV file (same pattern as others)
        self._wav_path = os.path.join(tempfile.gettempdir(), "speech.wav")
        self._wav_file: Optional[wave.Wave_write] = None
```

On `AudioChunk` event → write to WAV file
On `AudioStop` event → call model.transcribe()

### Model Wrapper

```python
class GlmAsrModel:
    def __init__(self, repo_id: str):
        self.processor = AutoProcessor.from_pretrained(repo_id)
        self.model = AutoModel.from_pretrained(repo_id, dtype=torch.bfloat16, device_map="auto")
        self.device = self.model.device  # Capture actual device

    def transcribe(self, wav_path: str, system_prompt: str) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "audio", "url": wav_path},
                {"type": "text", "text": system_prompt}
            ]
        }]

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        inputs = inputs.to(self.device, dtype=torch.bfloat16)

        outputs = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        return processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], 
                                     skip_special_tokens=True)[0]
```

### Key Differences from Transformers Whisper

| Aspect | TransformersWhisper | GlmAsr |
|--------|---------------------|--------|
| Model class | `AutoModelForSpeechSeq2Seq` | `AutoModel` |
| Dtype | default | `torch.bfloat16` |
| Device | manual | `"auto"` |
| Input format | raw audio tensor | audio URL path in chat template |
| System prompt | not supported | `--initial-prompt` becomes system text |
| Language param | used for forced decoder | ignored (GLM-ASR auto-detects) |

### Open Questions

1. **Audio format**: GLM-ASR expects an audio URL/path in the chat template. Do we need to convert WAV to a different format, or can we pass the temp file path directly?

### How Other Model Types Handle Async Inference

All existing model types follow an identical pattern:

1. **Lock-based serialization**: `asyncio.Lock` is created in `__main__.py` and passed to handlers
2. **Sync inference inside async context**: Direct sync calls wrapped in `async with self.model_lock:`
3. **No thread offloading**: They do NOT use `asyncio.to_thread()` or `run_in_executor()`

```python
# Pattern used by ALL handlers (faster_whisper, whisper_mps, transformers, kyutai-stt)
async with self.model_lock:
    result = self.model.transcribe(...)  # Sync call blocks event loop
```

**Recommendation**: Follow the same convention for GLM-ASR consistency. Use `asyncio.Lock` and call `transcribe()` synchronously inside the lock.
