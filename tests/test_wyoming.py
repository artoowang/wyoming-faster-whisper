import argparse
import asyncio
import time

from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info

from tests.audio_utils import send_audio_stream

DEFAULT_AUDIO_FILE = "play-recording.wav"


async def run() -> None:
    async with AsyncTcpClient(args.ip, int(args.port)) as client:
        await client.write_event(Describe().event())
        result = await client.read_event()
        assert result is not None
        assert Info.is_type(result.type)
        info = Info.from_event(result)
        print(info)

        transcript, execution_time = await send_audio_stream(
            client, args.audio_file, args.target_rate
        )

        print(transcript)
        print(f"execution_time: {execution_time}")


parser = argparse.ArgumentParser()
parser.add_argument("--ip", required=True)
parser.add_argument("--port", required=True)
parser.add_argument("--audio_file", default=DEFAULT_AUDIO_FILE)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--target_rate", default=None, type=int)
args = parser.parse_args()

asyncio.run(run())
