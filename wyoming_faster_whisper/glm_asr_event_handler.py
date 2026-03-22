"""Code for GLM-ASR transcription."""

import asyncio
import logging
import os
import time
import wave
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_SAMPLE_WIDTH = 2
_CHANNELS = 1


class GlmAsrModel:
    """Wrapper for GLM-ASR model."""

    def __init__(
        self,
        repo_id: str,
    ) -> None:
        """Initialize GLM-ASR model."""
        self.processor = AutoProcessor.from_pretrained(repo_id, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            repo_id,
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        # Store the actual device used by the model.
        self.device = self.model.device
        _LOGGER.info(f"GLM-ASR model loaded on device: {self.device}")

    def transcribe(self, audio_float: np.ndarray, system_prompt: str) -> str:
        """Returns transcription for audio array.

        Audio array must be float32 with values in [-1, 1], assumed 16Khz mono.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_float},
                    {"type": "text", "text": system_prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device, dtype=torch.bfloat16)
        input_length = inputs.input_ids.shape[1]

        # The official example does not do this: https://github.com/zai-org/GLM-ASR
        # But OpenCode suggests it.
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
            transcription = self.processor.batch_decode(
                outputs[:, input_length:], skip_special_tokens=True
            )[0]

        return transcription


class GlmAsrEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        wyoming_info: Info,
        model: GlmAsrModel,
        model_lock: asyncio.Lock,
        *args,
        audio_debug_dir: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_lock = model_lock
        self.initial_prompt = initial_prompt or ""
        self._audio_chunks: List[np.ndarray] = []

        self._wav_debug_dir = audio_debug_dir
        if self._wav_debug_dir is not None:
            # Ensure the debug directory exists
            os.makedirs(self._wav_debug_dir, exist_ok=True)

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            if chunk.rate != _SAMPLE_RATE:
                _LOGGER.error(
                    "Only supports 16kHz audio, but received rate %s", chunk.rate
                )
                return False
            if chunk.width != _SAMPLE_WIDTH:
                _LOGGER.error(
                    "Only supports 16-bit audio, but received width %s", chunk.width
                )
                return False
            if chunk.channels != _CHANNELS:
                _LOGGER.error(
                    "Only supports mono audio, but received %s channels",
                    chunk.channels,
                )
                return False

            audio_array = np.frombuffer(chunk.audio, dtype=np.int16)
            self._audio_chunks.append(audio_array)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug(
                "Audio stopped. Transcribing with prompt=%s", self.initial_prompt
            )

            if not self._audio_chunks:
                _LOGGER.error("No audio chunks received")
                return False

            audio_int16 = np.concatenate(self._audio_chunks)
            self._audio_chunks.clear()

            if self._wav_debug_dir is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                dst_path = os.path.join(self._wav_debug_dir, f"debug_{timestamp}.wav")
                with wave.open(dst_path, "wb") as wf:
                    wf.setnchannels(_CHANNELS)
                    wf.setsampwidth(_SAMPLE_WIDTH)
                    wf.setframerate(_SAMPLE_RATE)
                    wf.writeframes(audio_int16.tobytes())
                _LOGGER.debug("WAV debug copy written to %s", dst_path)

            audio_float = audio_int16.astype(np.float32) / 32768.0

            async with self.model_lock:
                text = self.model.transcribe(audio_float, self.initial_prompt)

            _LOGGER.info(text)

            await self.write_event(Transcript(text=text).event())
            _LOGGER.debug("Completed request")

            return False

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                _LOGGER.debug(
                    "Language hint received but ignored for GLM-ASR (auto-detects)"
                )
            # Clears the audio chunks in preparation for the AudioChunk events that will follow.
            self._audio_chunks.clear()
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True
