#!/usr/bin/env python3
import argparse
import asyncio
from pathlib import Path

from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info

from tests.audio_utils import send_audio_stream

AUDIO_DIR = Path("/Users/ollama/log/wyoming-stt-audio-debug/")


def parse_ground_truths(md_path: Path) -> list[tuple[str, str]]:
    entries = []
    content = md_path.read_text()
    lines = content.split("\n")
    in_table = False

    for line in lines:
        line = line.strip()
        if line.startswith("|") and "File Name" in line:
            in_table = True
            continue
        if in_table and line.startswith("|"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3 and parts[1] and not all(c in "-| " for c in line):
                filename = parts[1]
                ground_truth = parts[2]
                entries.append((filename, ground_truth))

    return entries


async def run_batch_tests(args: argparse.Namespace) -> list[tuple[str, str, str, float]]:
    entries = parse_ground_truths(Path("tests/ground_truths.md"))
    results: list[tuple[str, str, str, float]] = []
    target_rate = args.target_rate or 16000

    async with AsyncTcpClient(args.ip, int(args.port)) as client:
        await client.write_event(Describe().event())
        result = await client.read_event()
        assert result is not None
        assert Info.is_type(result.type)
        _info = Info.from_event(result)

    for i, (filename, ground_truth) in enumerate(entries, 1):
        audio_path = AUDIO_DIR / filename

        if not audio_path.exists():
            print(f"[!] {filename}: FILE NOT FOUND")
            results.append((filename, ground_truth, "<file not found>", 0.0))
            continue

        try:
            async with AsyncTcpClient(args.ip, int(args.port)) as client:
                transcription, exec_time = await send_audio_stream(
                    client, audio_path, target_rate
                )
            print(f"[{i}/{len(entries)}] {filename}")
            print(f"  Ground truth: {ground_truth}")
            print(f"  Transcription: {transcription}")
            print(f"  Time: {exec_time:.2f}s")
            print()
            results.append((filename, ground_truth, transcription, exec_time))

        except Exception as e:
            print(f"[!] [{i}/{len(entries)}] {filename}: ERROR - {e}")
            results.append((filename, ground_truth, f"<error: {e}>", 0.0))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch test STT against ground truths")
    parser.add_argument("--ip", default="localhost")
    parser.add_argument("--port", default="10301")
    parser.add_argument("--target_rate", type=int, default=None)
    args = parser.parse_args()

    print(f"Running batch tests against {args.ip}:{args.port}...\n")

    asyncio.run(run_batch_tests(args))


if __name__ == "__main__":
    main()
