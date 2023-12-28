"""Event handler for clients of the server."""
import argparse
import asyncio
import logging

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .faster_whisper import WhisperModel

import numpy as np
import whisper

_LOGGER = logging.getLogger(__name__)


class FasterWhisperEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model: WhisperModel,
        openai_whisper_model: whisper.Whisper,
        model_lock: asyncio.Lock,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.openai_whisper_model = openai_whisper_model
        self.model_lock = model_lock
        self.audio = bytes()
        self.audio_converter = AudioChunkConverter(
            rate=16000,
            width=2,
            channels=1,
        )
        self._language = self.cli_args.language

    async def handle_event(self, event: Event) -> bool:
        # _LOGGER.debug(f"Handle event: {event}")
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                self._language = transcribe.language
                _LOGGER.debug("Language set to %s", transcribe.language)
            return True

        if AudioChunk.is_type(event.type):
            if not self.audio:
                _LOGGER.debug("Receiving audio")

            chunk = AudioChunk.from_event(event)
            chunk = self.audio_converter.convert(chunk)
            self.audio += chunk.audio

            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped")
            async with self.model_lock:
                # segments, _info = self.model.transcribe(
                #     self.audio,
                #     beam_size=self.cli_args.beam_size,
                #     language=self._language,
                # )
                audio_samples = np.frombuffer(self.audio, dtype=np.int16)
                audio_samples_fp32 = audio_samples.astype(np.float32) / 32768.0
                result = whisper.transcribe(self.openai_whisper_model, audio_samples_fp32, language=self._language)

            # text = " ".join(segment.text for segment in segments)
            text = result["text"]
            _LOGGER.info(text)

            await self.write_event(Transcript(text=text).event())
            _LOGGER.debug("Completed request")

            # Reset
            self.audio = bytes()
            self._language = self.cli_args.language

            return False

        return True
