from typing import Generator
from wyoming.client import AsyncTcpClient
from wyoming.event import Event

import asyncio
import numpy as np
import pdb
import time
import whisper

WYOMING_IP = '127.0.0.1'
WYOMING_PORT = 10300
SAMPLE_RATE = 16000

def get_test_data() -> Generator[bytes, None, None]:
    CHUNK_LENGTH_IN_SECONDS = 1
    CHUNK_SAMPLES = CHUNK_LENGTH_IN_SECONDS * SAMPLE_RATE

    # audio_data_fp32 = whisper.load_audio('play-recording.wav')
    audio_data_fp32 = whisper.load_audio('Does the Past Still Exist.mp4')
    assert audio_data_fp32.dtype == np.float32
    assert len(audio_data_fp32.shape) == 1
    print(f'Loaded {audio_data_fp32.size} samples of audio data.')

    audio_data_int16 = (audio_data_fp32 * 32768.0).astype(np.int16)
    cur_sample = 0
    while cur_sample < audio_data_int16.size:
        end_sample = min(cur_sample + CHUNK_SAMPLES, audio_data_int16.size)
        audio_data = audio_data_int16[cur_sample:end_sample].tobytes()
        cur_sample = end_sample
        yield audio_data

async def run():
    async with AsyncTcpClient(WYOMING_IP, WYOMING_PORT) as client:
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

asyncio.run(run())