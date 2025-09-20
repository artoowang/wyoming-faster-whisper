from faster_whisper import WhisperModel
import time

model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("play-recording.wav", beam_size=5)

print("Detected language '%s' with probability %f" %
      (info.language, info.language_probability))

start = time.perf_counter()
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
end = time.perf_counter()
print(f"Transcription completed in {end - start:.2f} seconds")
