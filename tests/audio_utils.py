"""Shared audio utilities for Wyoming STT testing."""
from typing import Generator

import math

import numpy as np
from scipy.signal import resample_poly
from wave import Wave_read


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
