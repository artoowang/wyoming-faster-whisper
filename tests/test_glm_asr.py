import argparse
import pdb
import torch
from transformers import AutoModel, AutoProcessor

parser = argparse.ArgumentParser()
parser.add_argument("audio_file", help="Path to the audio file")
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

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "url": args.audio_file,
            },
            {
                "type": "text",
                "text": (
                    "You are a home assistant. Transcribe the user audio into commands. "
                    "Common device names include: morning scene, evening scene, night scene."
                    "Common commands include: turn on, turn off, open, close."
                ),
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
