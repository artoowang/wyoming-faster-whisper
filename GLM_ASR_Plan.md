# Plan: Add GLM-ASR Backend

## Files to Create/Modify

### 1. Create `wyoming_faster_whisper/glm_asr_event_handler.py`
- New `GlmAsrEventHandler` class (similar to existing handlers)
- Audio handling: 16kHz mono expected, accumulate chunks
- On `AudioStop`: transcribe using GLM-ASR
- Language handling (GLM-ASR uses chat template, so prompt goes in messages)

### 2. Modify `wyoming_faster_whisper/__main__.py`
- Add `--model-type glm-asr` option
- Add `glm-asr` case that loads model and runs `GlmAsrEventHandler`
- `--model` will be the HuggingFace repo_id (default: `zai-org/GLM-ASR-Nano-2512`)
- `--initial-prompt` can be passed as system message

### 3. Modify `tests/test_glm_asr.py` or create integration test
- Follow pattern from `test_faster_whisper.py`

## GLM-ASR Specific Considerations

| Aspect | Approach |
|--------|----------|
| Model loading | `AutoModel.from_pretrained(repo_id, dtype=torch.bfloat16, device_map="auto")` |
| Processor | `AutoProcessor.from_pretrained(repo_id)` |
| Audio format | 16kHz mono WAV |
| Transcription | Chat template with audio URL + system prompt |
| Device | Automatic (from `model.device`) |
| Initial prompt | Pass via chat template system message |

## Implementation Questions

1. Should `--initial-prompt` be the system message text, or do you want a separate `--glm-system-prompt` flag?
2. Should we support streaming (return partial results) or only final transcript?
3. Will GLM-ASR run on CPU/GPU or always use the automatic device mapping?

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

2. **System prompt**: Should `--initial-prompt` become the system text, or do you want a separate `--glm-system-prompt`?

3. **Async inference**: The current `test_glm_asr.py` runs synchronously. Should `transcribe()` be async to not block the event loop during generation?
