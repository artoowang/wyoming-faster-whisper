from typing import Optional
from wyoming.client import AsyncTcpClient
from wyoming.event import Event

import asyncio
import numpy as np
import pdb
import whisper

WYOMING_IP = '127.0.0.1'
WYOMING_PORT = 10300

async def run():
    async with AsyncTcpClient(WYOMING_IP, WYOMING_PORT) as client:
        await client.write_event(Event('describe'))
        result = await client.read_event()
        print(result)

        audio_data_fp32 = whisper.load_audio('play-recording.wav')
        assert audio_data_fp32.dtype == np.float32
        assert len(audio_data_fp32.shape) == 1
        audio_data = (audio_data_fp32 * 32768.0).astype(np.int16).tobytes()
        print(f'Loaded {len(audio_data)} bytes audio data.')

        await client.write_event(Event('transcibe', {'language': 'en'}))
        audio_metadata = {
            'rate': 16000,
            'width': 2,
            'channels': 1,
        }
        await client.write_event(Event('audio-start', audio_metadata))
        await client.write_event(Event('audio-chunk', audio_metadata, audio_data))
        await client.write_event(Event('audio-stop', audio_metadata))
        result = await client.read_event()
        print(result)

asyncio.run(run())

# TODO: Remove t he following

# def send_command(cmd: str, payload: Optional[bytes] = None):
#     wyoming_channel = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     wyoming_channel.connect((WYOMING_IP, WYOMING_PORT))
#     # TODO: check cmd does not end with newline.
#     wyoming_channel.send(cmd.encode('utf-8'))
#     if payload:
#         wyoming_channel.send(b'\n')
#         wyoming_channel.send(payload)
#         print(f'Sent {len(payload)} bytes.')
#     # time.sleep(5)
#     while True:
#         reply = wyoming_channel.recv(4096)
#         print(f'Received: {reply}')
#     # wyoming_channel.close()

# send_command('{"type":"describe"}\n{"type":"describe"}\n{"type": "transcibe", "data": {"language": "en"} }\n')
# send_command('{"type": "transcibe", "data": {"language": "en"} }')
# send_command('{"type": "audio-chunk", "data": {"rate": 16000, "width": 2, "channels": 1}, "payload_length": ' + str(len(audio_data)) + '}',
#              audio_data)
# send_command('{"type": "audio-stop"}')

# wyoming_channel = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# wyoming_channel.connect((WYOMING_IP, WYOMING_PORT))

# wyoming_channel.send('{"type":"describe"}'.encode('utf-8'))

# # "transcibe" is not a typo - the source code really expects that.
# wyoming_channel.send('{"type": "transcibe", "data": {"language": "en"} }\n'.encode('utf-8'))
# wyoming_channel.send(('{"type": "audio-chunk", "data": {"rate": 16000, "width": 2, "channels": 1}, "payload_length": ' + str(audio_data.size) + '}\n').encode('utf-8'))
# wyoming_channel.send(b'1234')
# # # wyoming_channel.send(audio_data.tobytes())
# wyoming_channel.send('{"type": "audio-stop"}\n'.encode('utf-8'))

# wyoming_channel.close()