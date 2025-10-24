from typing import Generator
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info

import argparse
import asyncio
import math
import numpy as np
import time
import wave
from wave import Wave_read
from scipy.signal import resample_poly

DEFAULT_AUDIO_FILE = 'play-recording.wav'


def get_test_data(wav: Wave_read) -> Generator[bytes, None, None]:
    n_frames = wav.getnframes()
    audio_bytes = wav.readframes(n_frames)
    assert wav.getsampwidth() == 2, 'Expected 16-bit audio'
    audio_data_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    print(f'Loaded {audio_data_int16.size} samples of audio data.')

    if args.target_rate is not None and wav.getframerate() != args.target_rate:
        print(f'Resample from {wav.getframerate()}Hz to {args.target_rate}Hz ...')
        # reduce ratio to small integers to speed up/filter design
        g = math.gcd(args.target_rate, wav.getframerate())
        up = args.target_rate // g
        down = wav.getframerate() // g

        # Work in float for filtering; resample_poly accepts ints too but float is explicit
        audio_float = audio_data_int16.astype(np.float32)
        audio_resampled = resample_poly(audio_float, up, down)

        audio_resampled = np.clip(audio_resampled, -32768, 32767)
        audio_data_int16 = np.round(audio_resampled).astype(np.int16)
        print(f'Resampled to {audio_data_int16.size} samples.')

    CHUNK_LENGTH_IN_SECONDS = 1
    CHUNK_SAMPLES = CHUNK_LENGTH_IN_SECONDS * args.target_rate

    cur_sample = 0
    while cur_sample < audio_data_int16.size:
        end_sample = min(cur_sample + CHUNK_SAMPLES, audio_data_int16.size)
        audio_data = audio_data_int16[cur_sample:end_sample].tobytes()
        cur_sample = end_sample
        yield audio_data


async def run():
    async with AsyncTcpClient(args.ip, int(args.port)) as client:
        await client.write_event(Describe().event())
        result = await client.read_event()
        assert Info.is_type(result.type)
        info = Info.from_event(result)
        print(info)

        with wave.open(args.audio_file, "rb") as wav:
            await client.write_event(Transcribe().event())
            if args.target_rate is None:
                args.target_rate = wav.getframerate()
            audio_metadata = {
                'rate': args.target_rate,
                'width': wav.getsampwidth(),
                'channels': wav.getnchannels(),
            }
            print(f'Audio metadata: {audio_metadata}')
            await client.write_event(AudioStart(**audio_metadata).event())
            for audio_data in get_test_data(wav):
                print(f'Write {len(audio_data)} bytes.')
                await client.write_event(AudioChunk(**audio_metadata, audio=audio_data).event())
            await client.write_event(AudioStop().event())

            start_time = time.time()
            result = await client.read_event()
            assert Transcript.is_type(result.type)
            transcript = Transcript.from_event(result)
            execution_time = time.time() - start_time

            print(transcript)
            print(f'execution_time: {execution_time}')

parser = argparse.ArgumentParser()
parser.add_argument("--ip", required=True)
parser.add_argument("--port", required=True)
parser.add_argument("--audio_file", default=DEFAULT_AUDIO_FILE)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--target_rate", default=None, type=int)
args = parser.parse_args()

asyncio.run(run())
