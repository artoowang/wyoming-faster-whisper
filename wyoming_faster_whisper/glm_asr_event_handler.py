"""Code for GLM-ASR transcription."""

import asyncio
import logging
import os
import tempfile
import wave
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import AutoModel, AutoProcessor
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)


class GlmAsrModel:
    """Wrapper for GLM-ASR model."""

    def __init__(
        self,
        repo_id: str,
        cache_dir: Optional[Union[str, Path]] = None,
        local_files_only: bool = False,
    ) -> None:
        """Initialize GLM-ASR model."""
        self.processor = AutoProcessor.from_pretrained(
            repo_id, cache_dir=cache_dir, local_files_only=local_files_only
        )
        self.model = AutoModel.from_pretrained(
            repo_id,
            dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.model.eval()
        self.device = self.model.device

    def transcribe(self, wav_path: str, system_prompt: str) -> str:
        """Returns transcription for WAV file.

        WAV file must be 16Khz 16-bit mono audio.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "url": wav_path},
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
        initial_prompt: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_lock = model_lock
        self.initial_prompt = initial_prompt or ""
        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = os.path.join(self._wav_dir.name, "speech.wav")
        self._wav_file: Optional[wave.Wave_write] = None

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            if self._wav_file is None:
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)

            self._wav_file.writeframes(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug(
                "Audio stopped. Transcribing with prompt=%s", self.initial_prompt
            )
            assert self._wav_file is not None

            self._wav_file.close()
            self._wav_file = None

            async with self.model_lock:
                text = self.model.transcribe(self._wav_path, self.initial_prompt)

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
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True
