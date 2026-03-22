import argparse
import math
import wave
import torch
import numpy as np
from scipy.signal import resample_poly
from transformers import AutoModel, AutoProcessor

parser = argparse.ArgumentParser()
parser.add_argument("audio_file", help="Path to the audio file")
parser.add_argument(
    "--prompt",
    default=(
        "You are a home assistant. Transcribe the user audio into commands. "
        "Common device names include: morning scene, evening scene, night scene. "
        "Common commands include: turn on, turn off, open, close."
    ),
    help="Prompt to guide transcription",
)
args = parser.parse_args()

# The following code is adapted from
# https://github.com/zai-org/GLM-ASR?tab=readme-ov-file#example-code

repo_id = "zai-org/GLM-ASR-Nano-2512"
processor = AutoProcessor.from_pretrained(repo_id)
device = "auto"
model = AutoModel.from_pretrained(repo_id, dtype=torch.bfloat16, device_map=device)
# Update the device variable to the actual device used by the model.
device = model.device
print(f"Device type: {device}")

audio_float = None
with wave.open(args.audio_file, "rb") as wav:
    wav_rate = wav.getframerate()
    n_frames = wav.getnframes()
    audio_bytes = wav.readframes(n_frames)
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0

    if wav_rate != 16000:
        print(f"Resampling from {wav_rate}Hz to 16000Hz")
        g = math.gcd(16000, wav_rate)
        up, down = 16000 // g, wav_rate // g
        audio_float = resample_poly(audio_float, up, down)

assert audio_float is not None, "Failed to load audio"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "audio": audio_float,
            },
            {
                "type": "text",
                "text": args.prompt,
            },
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = inputs.to(device, dtype=torch.bfloat16)
input_length = inputs.input_ids.shape[1]
outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
outputs_decoded = processor.batch_decode(
    outputs[:, input_length:], skip_special_tokens=True
)
print(outputs_decoded)
