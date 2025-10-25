"""Event handler for clients of the server."""
import argparse
import asyncio
import logging
from typing import Optional
import mlx.core as mx
import numpy as np
import os
import time
import wave

from whisper_mps import whisper

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)


class WhisperMpsEventHandler(AsyncEventHandler):
    """Event handler for whisper-mps."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model_lock: asyncio.Lock,
        *args,
        initial_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model_lock = model_lock
        self.initial_prompt = initial_prompt
        self._language = self.cli_args.language
        # Store audio chunks and metadata
        self._audio_chunks: list[np.ndarray] = []
        self._sample_rate: Optional[int] = None
        self._sample_width: Optional[int] = None
        self._channels: Optional[int] = None
        self._wav_debug_dir = self.cli_args.audio_debug_dir

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            # Store metadata from first chunk
            if not self._audio_chunks:
                assert chunk.rate == 16000, f"Only supports 16kHz audio, but received rate {chunk.rate}"
                assert chunk.width == 2, f"Only supports 16-bit audio, but received width {chunk.width}"
                assert chunk.channels == 1, f"Only supports mono audio, but received {chunk.channels} channels"
                self._sample_rate = chunk.rate
                self._sample_width = chunk.width
                self._channels = chunk.channels

            # Convert bytes to int16 numpy array and store
            audio_array = np.frombuffer(chunk.audio, dtype=np.int16)
            self._audio_chunks.append(audio_array)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug(
                "Audio stopped. Transcribing with initial prompt=%s",
                self.initial_prompt,
            )
            assert self._audio_chunks, "No audio chunks received"

            # Concatenate all chunks and normalize to float16 [-1, 1]
            audio_int16 = np.concatenate(self._audio_chunks)
            audio = audio_int16.astype(np.float16) / 32768.0  # int16 range is [-32768, 32767]

            # Convert to MLX array with shape (# of samples,)
            audio = mx.array(audio)

            # Clear chunks for next request
            self._audio_chunks.clear()
            self._sample_rate = None
            self._sample_width = None
            self._channels = None

            if self._wav_debug_dir is not None:
                try:
                    # Ensure the debug directory exists
                    os.makedirs(self._wav_debug_dir, exist_ok=True)

                    # Timestamp suffix with microseconds to avoid collisions
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    dst_name = f"debug_{timestamp}.wav"
                    dst_path = os.path.join(self._wav_debug_dir, dst_name)

                    # Write WAV file
                    with wave.open(dst_path, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(16000)
                        wav_file.writeframes(audio_int16.tobytes())

                    _LOGGER.debug("WAV debug copy written to %s", dst_path)
                except Exception as exc:
                    _LOGGER.exception("Failed to write WAV debug copy: %s", exc)

            start_time = time.time()
            async with self.model_lock:
                result = whisper.transcribe(audio=audio, model=self.cli_args.model, language=self._language)
            _LOGGER.debug("Transcription completed in %.2f seconds", time.time() - start_time)
            _LOGGER.debug(f"Result: {result}")

            await self.write_event(Transcript(text=result["text"]).event())
            _LOGGER.debug("Completed request")

            # Reset
            self._language = self.cli_args.language

            return False

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                self._language = transcribe.language
                _LOGGER.debug("Language set to %s", transcribe.language)
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True
