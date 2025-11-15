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
import subprocess
import tempfile

from whisper_mps import whisper

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)
_SAMPLE_RATE = 16000
_SAMPLE_WIDTH = 2
_CHANNELS = 1


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
        self._wav_debug_dir = self.cli_args.audio_debug_dir

        if self._wav_debug_dir is not None:
            # Ensure the debug directory exists
            os.makedirs(self._wav_debug_dir, exist_ok=True)

    def _ffmpeg_arnndn_denoise(self, audio_int16: np.ndarray) -> np.ndarray:
        """Run ffmpeg arnndn denoiser on `audio_int16` and return denoised int16 numpy array.

        This writes input to a temporary WAV, calls ffmpeg to apply `arnndn`, reads the
        output WAV back, and returns the samples as np.int16.
        """
        if self._wav_debug_dir is not None:
            # Timestamp suffix with microseconds to avoid collisions
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            in_name = os.path.join(self._wav_debug_dir, f"debug_{timestamp}.wav")
            out_name = os.path.join(self._wav_debug_dir, f"debug_denoised_{timestamp}.wav")
        else:
            in_name = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            out_name = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

        try:
            # Write input WAV
            with wave.open(in_name, "wb") as wf:
                wf.setnchannels(_CHANNELS)
                wf.setsampwidth(_SAMPLE_WIDTH)
                wf.setframerate(_SAMPLE_RATE)
                wf.writeframes(audio_int16.tobytes())

            # Run ffmpeg arnndn filter
            cmd = [
                "/opt/homebrew/bin/ffmpeg",
                "-y",
                "-hide_banner",
                "-i",
                in_name,
                "-af",
                "arnndn=m=/Users/ollama/Programs/wyoming-faster-whisper/rnnn/std.rnnn",
                "-ar",
                f"{_SAMPLE_RATE}",
                out_name,
            ]
            subprocess.run(cmd, check=True)

            # Read output WAV
            with wave.open(out_name, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                denoised = np.frombuffer(frames, dtype=np.int16)

            return denoised
        finally:
            if self._wav_debug_dir is None:
                try:
                    os.unlink(in_name)
                except Exception:
                    pass
                try:
                    os.unlink(out_name)
                except Exception:
                    pass

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            # Store metadata from first chunk
            if not self._audio_chunks:
                assert chunk.rate == _SAMPLE_RATE, f"Only supports 16kHz audio, but received rate {chunk.rate}"
                assert chunk.width == _SAMPLE_WIDTH, f"Only supports 16-bit audio, but received width {chunk.width}"
                assert chunk.channels == _CHANNELS, f"Only supports mono audio, but received {chunk.channels} channels"

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

            # Concatenate all chunks
            audio_int16 = np.concatenate(self._audio_chunks)

            # Optionally denoise with ffmpeg arnndn if enabled in CLI args
            if self.cli_args.ffmpeg_denoise:
                try:
                    audio_int16 = self._ffmpeg_arnndn_denoise(audio_int16)
                    _LOGGER.debug("Audio denoised with ffmpeg arnndn")
                except Exception:
                    _LOGGER.exception("FFmpeg denoising failed; proceeding with original audio")
            else:
                if self._wav_debug_dir is not None:
                    try:
                        # Timestamp suffix with microseconds to avoid collisions
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        dst_path = os.path.join(self._wav_debug_dir, f"debug_{timestamp}.wav")

                        # Write WAV file
                        with wave.open(dst_path, 'wb') as wav_file:
                            wav_file.setnchannels(_CHANNELS)
                            wav_file.setsampwidth(_SAMPLE_WIDTH)
                            wav_file.setframerate(_SAMPLE_RATE)
                            wav_file.writeframes(audio_int16.tobytes())

                        _LOGGER.debug("WAV debug copy written to %s", dst_path)
                    except Exception as exc:
                        _LOGGER.exception("Failed to write WAV debug copy: %s", exc)

            # Normalize to float16 [-1, 1]
            audio = audio_int16.astype(np.float16) / 32768.0  # int16 range is [-32768, 32767]

            # Convert to MLX array with shape (# of samples,)
            audio = mx.array(audio)

            # Clear chunks for next request
            self._audio_chunks.clear()

            start_time = time.time()
            async with self.model_lock:
                result = whisper.transcribe(
                    audio=audio,
                    model=self.cli_args.model,
                    language=self._language,
                    initial_prompt=self.initial_prompt)
            _LOGGER.debug("Transcription completed in %.2f seconds", time.time() - start_time)
            _LOGGER.debug(f"Result: {result}")

            await self.write_event(Transcript(text=result["text"], language=result["language"]).event())
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
