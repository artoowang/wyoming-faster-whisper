#!/usr/bin/env python3
import argparse
import asyncio
import csv
import difflib
import math
import re
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import resample_poly
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info

AUDIO_DIR = Path("/Users/ollama/log/wyoming-stt-audio-debug/")


@dataclass
class GroundTruthEntry:
    filename: str
    ground_truth: str
    category: str
    notes: str = ""


@dataclass
class TestResult:
    filename: str
    ground_truth: str
    transcription: str
    exact_match: bool
    wer_score: float
    category: str
    notes: str = ""
    execution_time: float = 0.0


def parse_ground_truths(md_path: Path) -> list[GroundTruthEntry]:
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
            if len(parts) >= 3 and parts[1]:
                filename = parts[1]
                ground_truth = parts[2]

                if ground_truth.startswith("<"):
                    if "empty" in ground_truth.lower():
                        category = "empty"
                    else:
                        category = "noisy"
                    notes = ground_truth
                else:
                    category = "clear"
                    notes = ""

                entries.append(
                    GroundTruthEntry(
                        filename=filename,
                        ground_truth=ground_truth,
                        category=category,
                        notes=notes,
                    )
                )

    return entries


def calculate_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    levenshtein_distance = d[len(ref_words)][len(hyp_words)]
    wer = levenshtein_distance / len(ref_words)
    return wer


def is_exact_match(ref: str, hyp: str) -> bool:
    return ref.strip().lower() == hyp.strip().lower()


def check_transcription_quality(
    transcription: str, category: str, ground_truth: str
) -> tuple[bool, float]:
    transcription = transcription.strip()

    if category == "empty":
        is_pass = len(transcription) < 5
        wer = 1.0 if transcription else 0.0
        return is_pass, wer

    if category == "noisy":
        is_pass = True
        wer = calculate_wer(
            ground_truth.replace(f" <{category}>", "").strip(), transcription
        )
        return is_pass, wer

    exact = is_exact_match(ground_truth, transcription)
    wer = calculate_wer(ground_truth, transcription)
    is_pass = exact or wer < 0.10

    return is_pass, wer


async def test_single_file(
    ip: str,
    port: int,
    target_rate: int,
    audio_path: Path,
) -> tuple[str, float]:
    start_time = asyncio.get_event_loop().time()

    async with AsyncTcpClient(ip, port) as client:
        with wave.open(str(audio_path), "rb") as wav:
            await client.write_event(Transcribe().event())

            wav_rate = wav.getframerate()
            wav_width = wav.getsampwidth()
            wav_channels = wav.getnchannels()

            if wav_rate != target_rate:
                n_frames = wav.getnframes()
                audio_bytes = wav.readframes(n_frames)
                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                g = math.gcd(target_rate, wav_rate)
                up, down = target_rate // g, wav_rate // g
                audio_float = audio_int16.astype(np.float32)
                audio_resampled = resample_poly(audio_float, up, down)
                audio_resampled = np.clip(audio_resampled, -32768, 32767)
                audio_int16 = np.round(audio_resampled).astype(np.int16)
            else:
                n_frames = wav.getnframes()
                audio_bytes = wav.readframes(n_frames)
                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

            audio_metadata = {
                "rate": target_rate,
                "width": wav_width,
                "channels": wav_channels,
            }
            await client.write_event(AudioStart(**audio_metadata).event())

            CHUNK_LENGTH_IN_SECONDS = 1
            CHUNK_SAMPLES = CHUNK_LENGTH_IN_SECONDS * target_rate
            cur_sample = 0

            while cur_sample < audio_int16.size:
                end_sample = min(cur_sample + CHUNK_SAMPLES, audio_int16.size)
                audio_data = audio_int16[cur_sample:end_sample].tobytes()
                cur_sample = end_sample
                await client.write_event(
                    AudioChunk(**audio_metadata, audio=audio_data).event()
                )

            await client.write_event(AudioStop().event())

            result = await client.read_event()
            assert Transcript.is_type(result.type)
            transcript = Transcript.from_event(result)
            execution_time = asyncio.get_event_loop().time() - start_time

            return transcript.text, execution_time


async def run_batch_tests(args: argparse.Namespace) -> list[TestResult]:
    entries = parse_ground_truths(Path("tests/ground_truths.md"))

    results: list[TestResult] = []
    target_rate = args.target_rate or 16000

    async with AsyncTcpClient(args.ip, int(args.port)) as client:
        await client.write_event(Describe().event())
        result = await client.read_event()
        assert Info.is_type(result.type)
        _info = Info.from_event(result)

    for i, entry in enumerate(entries, 1):
        audio_path = AUDIO_DIR / entry.filename

        if not audio_path.exists():
            print(f"[!] {entry.filename}: FILE NOT FOUND")
            results.append(
                TestResult(
                    filename=entry.filename,
                    ground_truth=entry.ground_truth,
                    transcription="<file not found>",
                    exact_match=False,
                    wer_score=1.0,
                    category=entry.category,
                    notes=f"File not found at {audio_path}",
                )
            )
            continue

        try:
            transcription, exec_time = await test_single_file(
                args.ip, int(args.port), target_rate, audio_path
            )
            is_pass, wer = check_transcription_quality(
                transcription, entry.category, entry.ground_truth
            )
            exact = is_exact_match(entry.ground_truth, transcription)

            symbol = "✓" if (exact or (entry.category == "noisy")) else "✗"
            print(
                f'[{symbol}] [{i}/{len(entries)}] {entry.filename}: "{transcription}"'
            )

            results.append(
                TestResult(
                    filename=entry.filename,
                    ground_truth=entry.ground_truth,
                    transcription=transcription,
                    exact_match=exact,
                    wer_score=wer,
                    category=entry.category,
                    notes=entry.notes,
                    execution_time=exec_time,
                )
            )

        except Exception as e:
            print(f"[!] [{i}/{len(entries)}] {entry.filename}: ERROR - {e}")
            results.append(
                TestResult(
                    filename=entry.filename,
                    ground_truth=entry.ground_truth,
                    transcription=f"<error: {e}>",
                    exact_match=False,
                    wer_score=1.0,
                    category=entry.category,
                    notes=str(e),
                )
            )

    return results


def generate_csv_report(results: list[TestResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "filename",
                "ground_truth",
                "transcription",
                "exact_match",
                "wer_score",
                "category",
                "notes",
                "execution_time",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.filename,
                    r.ground_truth,
                    r.transcription,
                    "Yes" if r.exact_match else "No",
                    f"{r.wer_score:.4f}",
                    r.category,
                    r.notes,
                    f"{r.execution_time:.2f}",
                ]
            )


def print_summary(results: list[TestResult]) -> None:
    clear_results = [r for r in results if r.category == "clear"]
    empty_results = [r for r in results if r.category == "empty"]
    noisy_results = [r for r in results if r.category == "noisy"]

    clear_exact = sum(1 for r in clear_results if r.exact_match)
    clear_acceptable = sum(
        1 for r in clear_results if r.exact_match or r.wer_score < 0.10
    )
    empty_pass = sum(1 for r in empty_results if len(r.transcription.strip()) < 5)

    total_time = sum(r.execution_time for r in results)
    avg_time = total_time / len(results) if results else 0

    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)

    print(f"\nTotal Tests: {len(results)}")
    print(f"Total Execution Time: {total_time:.2f}s (avg: {avg_time:.2f}s per file)")

    print(f"\n--- Clear Commands ({len(clear_results)}) ---")
    print(f"Exact Match: {clear_exact} ({clear_exact/len(clear_results)*100:.1f}%)")
    print(
        f"Acceptable (WER<10%): {clear_acceptable} ({clear_acceptable/len(clear_results)*100:.1f}%)"
    )
    print(f"Failed: {len(clear_results) - clear_acceptable}")

    print(f"\n--- Empty/Noise Tests ({len(empty_results) + len(noisy_results)}) ---")
    print(f"Empty Tests (expected no/low output): {len(empty_results)}")
    print(f"Noisy/Low-Volume Tests: {len(noisy_results)}")

    print("\n--- Failed Clear Commands ---")
    failed = [r for r in clear_results if not (r.exact_match or r.wer_score < 0.10)]
    if failed:
        for r in failed:
            print(f"  {r.filename}")
            print(f'    Expected: "{r.ground_truth}"')
            print(f'    Got:      "{r.transcription}"')
            print(f"    WER: {r.wer_score:.2%}")
    else:
        print("  None!")

    print("\n--- Expected Poor Quality (Reference Only) ---")
    if noisy_results:
        for r in noisy_results:
            print(f"  {r.filename}")
            print(f'    Expected: "{r.ground_truth}"')
            print(f'    Got:      "{r.transcription}"')
            print(f"    WER: {r.wer_score:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch test STT against ground truths")
    parser.add_argument("--ip", default="localhost")
    parser.add_argument("--port", default="10301")
    parser.add_argument("--target_rate", type=int, default=None)
    parser.add_argument("--output_dir", default="tests/results")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(args.output_dir) / f"ground_truth_results_{timestamp}.csv"

    print(f"Running batch tests against {args.ip}:{args.port}...")
    print(f"Results will be saved to: {csv_path}\n")

    results = asyncio.run(run_batch_tests(args))
    generate_csv_report(results, csv_path)
    print_summary(results)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
