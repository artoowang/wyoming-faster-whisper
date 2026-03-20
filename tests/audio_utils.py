"""Shared audio utilities for Wyoming STT testing."""

from __future__ import annotations

import asyncio
import math
import wave
from pathlib import Path
from typing import Generator

import numpy as np
from scipy.signal import resample_poly
from wave import Wave_read
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient


def get_audio_chunks(wav: Wave_read, target_rate: int) -> Generator[bytes, None, None]:
    """Read audio from wav file, resample if needed, yield chunks.

    Args:
        wav: Open wave file (read mode)
        target_rate: Target sample rate (e.g., 16000)

    Yields:
        Audio chunks as bytes (16-bit PCM)
    """
    n_frames = wav.getnframes()
    audio_bytes = wav.readframes(n_frames)
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

    wav_rate = wav.getframerate()
    if wav_rate != target_rate:
        g = math.gcd(target_rate, wav_rate)
        up, down = target_rate // g, wav_rate // g
        audio_float = audio_int16.astype(np.float32)
        audio_resampled = resample_poly(audio_float, up, down)
        audio_resampled = np.clip(audio_resampled, -32768, 32767)
        audio_int16 = np.round(audio_resampled).astype(np.int16)

    chunk_samples = 1 * target_rate  # 1 second chunks
    cur_sample = 0
    while cur_sample < audio_int16.size:
        end_sample = min(cur_sample + chunk_samples, audio_int16.size)
        yield audio_int16[cur_sample:end_sample].tobytes()
        cur_sample = end_sample


async def send_audio_stream(
    client: AsyncTcpClient,
    wav_path: str | Path,
    target_rate: int | None = None,
) -> tuple[str, float]:
    """Send audio file to STT server and return transcript + timing.

    Args:
        client: Connected AsyncTcpClient
        wav_path: Path to WAV audio file
        target_rate: Target sample rate (e.g., 16000). If None, uses the WAV file's
            native rate.

    Returns:
        Tuple of (transcript text, execution time in seconds)
    """
    with wave.open(str(wav_path), "rb") as wav:
        wav_rate = wav.getframerate()
        effective_rate = target_rate if target_rate is not None else wav_rate
        audio_metadata = {
            "rate": effective_rate,
            "width": wav.getsampwidth(),
            "channels": wav.getnchannels(),
        }
        await client.write_event(Transcribe().event())
        await client.write_event(AudioStart(**audio_metadata).event())
        for audio_data in get_audio_chunks(wav, effective_rate):
            await client.write_event(
                AudioChunk(**audio_metadata, audio=audio_data).event()
            )
        await client.write_event(AudioStop().event())

    start_time = asyncio.get_event_loop().time()
    result = await client.read_event()
    assert result is not None
    assert Transcript.is_type(result.type)
    transcript = Transcript.from_event(result)
    execution_time = asyncio.get_event_loop().time() - start_time

    return transcript.text, execution_time
