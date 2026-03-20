import argparse
import asyncio
import time
import wave

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info

from tests.audio_utils import get_audio_chunks

DEFAULT_AUDIO_FILE = "play-recording.wav"


async def run() -> None:
    async with AsyncTcpClient(args.ip, int(args.port)) as client:
        await client.write_event(Describe().event())
        result = await client.read_event()
        assert result is not None
        assert Info.is_type(result.type)
        info = Info.from_event(result)
        print(info)

        with wave.open(args.audio_file, "rb") as wav:
            await client.write_event(Transcribe().event())
            if args.target_rate is None:
                args.target_rate = wav.getframerate()
            audio_metadata = {
                "rate": args.target_rate,
                "width": wav.getsampwidth(),
                "channels": wav.getnchannels(),
            }
            print(f"Audio metadata: {audio_metadata}")
            await client.write_event(AudioStart(**audio_metadata).event())
            for audio_data in get_audio_chunks(wav, args.target_rate):
                print(f"Write {len(audio_data)} bytes.")
                await client.write_event(
                    AudioChunk(**audio_metadata, audio=audio_data).event()
                )
            await client.write_event(AudioStop().event())

            start_time = time.time()
            result = await client.read_event()
            assert result is not None
            assert Transcript.is_type(result.type)
            transcript = Transcript.from_event(result)
            execution_time = time.time() - start_time

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
