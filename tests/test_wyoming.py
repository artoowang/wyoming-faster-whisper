from typing import Generator
from wyoming.client import AsyncTcpClient
from wyoming.event import Event

import argparse
import asyncio
import numpy as np
import time
import wave

DEFAULT_AUDIO_FILE = 'play-recording.wav'
SAMPLE_RATE = 16000

def get_test_data() -> Generator[bytes, None, None]:
    CHUNK_LENGTH_IN_SECONDS = 1
    CHUNK_SAMPLES = CHUNK_LENGTH_IN_SECONDS * SAMPLE_RATE

    with wave.open(args.audio_file, "rb") as wav:
        n_frames = wav.getnframes()
        audio_bytes = wav.readframes(n_frames)
        audio_data_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        print(f'Loaded {audio_data_int16.size} samples of audio data.')

        cur_sample = 0
        while cur_sample < audio_data_int16.size:
            end_sample = min(cur_sample + CHUNK_SAMPLES, audio_data_int16.size)
            audio_data = audio_data_int16[cur_sample:end_sample].tobytes()
            cur_sample = end_sample
            yield audio_data

async def run():
    async with AsyncTcpClient(args.ip, int(args.port)) as client:
        await client.write_event(Event('describe'))
        result = await client.read_event()
        print(result)

        await client.write_event(Event('transcibe', {'language': 'en'}))
        audio_metadata = {
            'rate': SAMPLE_RATE,
            'width': 2,
            'channels': 1,
        }
        await client.write_event(Event('audio-start', audio_metadata))
        for audio_data in get_test_data():
            print(f'Write {len(audio_data)} bytes.')
            await client.write_event(Event('audio-chunk', audio_metadata, audio_data))
        await client.write_event(Event('audio-stop', audio_metadata))
        start_time = time.time()
        result = await client.read_event()
        execution_time = time.time() - start_time
        print(result)
        print(f'execution_time: {execution_time}')

parser = argparse.ArgumentParser()
parser.add_argument("--ip", required=True)
parser.add_argument("--port", required=True)
parser.add_argument("--audio_file", default=DEFAULT_AUDIO_FILE)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

asyncio.run(run())
