"""Event handler for clients of the server."""
import argparse
import asyncio
import json
import logging
import os
import tempfile
import time
import wave
import shutil
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece
import sphn
from huggingface_hub import hf_hub_download
from moshi_mlx import models, utils
from scipy.signal import resample

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)
# TODO: Can we support other sample rates?
_SAMPLE_RATE = 24000


class KyutaiSttModel:
    """Wrapper for Kyutai STT model."""

    def __init__(self, hf_repo: str) -> None:
        lm_config = hf_hub_download(hf_repo, "config.json")
        with open(lm_config, "r") as fobj:
            lm_config = json.load(fobj)
        _LOGGER.info(f"lm_config:\n{lm_config}")
        self._stt_config = lm_config.get("stt_config", None)

        mimi_weights = hf_hub_download(hf_repo, lm_config["mimi_name"])
        moshi_name = lm_config.get("moshi_name", "model.safetensors")
        moshi_weights = hf_hub_download(hf_repo, moshi_name)
        text_tokenizer = hf_hub_download(hf_repo, lm_config["tokenizer_name"])

        lm_config = models.LmConfig.from_config_dict(lm_config)
        model = models.Lm(lm_config)
        model.set_dtype(mx.bfloat16)
        if moshi_weights.endswith(".q4.safetensors"):
            nn.quantize(model, bits=4, group_size=32)
        elif moshi_weights.endswith(".q8.safetensors"):
            nn.quantize(model, bits=8, group_size=64)

        _LOGGER.info(f"loading model weights from {moshi_weights}")
        if hf_repo.endswith("-candle"):
            model.load_pytorch_weights(moshi_weights, lm_config, strict=True)
        else:
            model.load_weights(moshi_weights, strict=True)

        _LOGGER.info(f"loading the text tokenizer from {text_tokenizer}")
        text_tokenizer = sentencepiece.SentencePieceProcessor(text_tokenizer)  # type: ignore

        _LOGGER.info(f"loading the audio tokenizer {mimi_weights}")
        audio_tokenizer = models.mimi.Mimi(models.mimi_202407(32))
        audio_tokenizer.load_pytorch_weights(str(mimi_weights), strict=True)

        # Copied from moshi_mlx.run_inference.
        if model.condition_provider is not None:
            ct = model.condition_provider.condition_tensor("description", "very_good")
        else:
            ct = None

        _LOGGER.info("warming up the model")
        model.warmup(ct)
        _LOGGER.info("done warming up the model")

        self._model = model
        self._audio_tokenizer = audio_tokenizer
        self._text_tokenizer = text_tokenizer


    def transcribe(self, audio_path: str) -> str:
        audio, _ = sphn.read(audio_path, sample_rate=_SAMPLE_RATE)
        assert audio.ndim == 2 and audio.shape[0] == 1, "Expected mono audio"

        if self._stt_config is not None:
            pad_left = self._stt_config.get("audio_silence_prefix_seconds", 0.0)
            pad_left = int(pad_left * _SAMPLE_RATE)
            pad_right = self._stt_config.get("audio_delay_seconds", 0.0)
            pad_right = int((pad_right + 1.0) * _SAMPLE_RATE)
            audio = mx.concat([
                mx.zeros((1, pad_left)),
                mx.array(audio),
                mx.zeros((1, pad_right)),
            ], axis=-1)
        else:
            # This was the original padding used, not sure why.
            audio = mx.concat([mx.array(audio), mx.zeros((1, 48000))], axis=-1)

        steps = audio.shape[-1] // 1920
        gen = models.LmGen(
            model=self._model,
            max_steps=steps,
            text_sampler=utils.Sampler(top_k=25, temp=0),
            audio_sampler=utils.Sampler(top_k=250, temp=0.8),
            check=False,
        )

        _LOGGER.info(f"starting inference (audio.shape: {audio.shape})")
        start_time = time.time()
        text = ""
        for start_idx in range(0, steps * 1920, 1920):
            block = audio[:, None, start_idx : start_idx + 1920]
            other_audio_tokens = self._audio_tokenizer.encode_step(block).transpose(0, 2, 1)
            text_token = gen.step(other_audio_tokens[0])
            text_token = text_token[0].item()
            if text_token not in (0, 3):
                text_piece = self._text_tokenizer.id_to_piece(text_token)  # type: ignore
                text_piece = text_piece.replace("▁", " ")
                text += text_piece
        inference_seconds = time.time() - start_time
        steps_per_second = steps / (time.time() - start_time)
        _LOGGER.info(f"inference time: {inference_seconds}s, steps: {steps}, steps per sec: {steps_per_second}")

        return text


class KyutaiSttEventHandler(AsyncEventHandler):
    """Event handler for using Kyutai STT.

    https://kyutai.org/next/stt
    """

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model: KyutaiSttModel,
        model_lock: asyncio.Lock,
        *args,
        initial_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_lock = model_lock
        # TODO: How to use initial prompt?
        self.initial_prompt = initial_prompt
        # TODO: Current only support English so this is ignored.
        self._language = self.cli_args.language
        # TODO: In theory we can support streaming, but it will require some careful handling of the incoming chunks.
        # For now we just wait until the audio stops.
        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = os.path.join(self._wav_dir.name, "speech.wav")
        self._wav_file: Optional[wave.Wave_write] = None
        self._chunk_sample_rate: Optional[int] = None
        self._wav_debug_dir = self.cli_args.audio_debug_dir


    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            if self._wav_file is None:
                assert chunk.width == 2, "Only supports 16-bit audio, but received width {chunk.width}"
                assert chunk.channels == 1, "Only supports mono audio, but received {chunk.channels}"
                self._chunk_sample_rate = chunk.rate
                _LOGGER.debug(
                    "Starting %s with rate=%d -> %d",
                    self._wav_path,
                    self._chunk_sample_rate,
                    _SAMPLE_RATE,
                )
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(_SAMPLE_RATE)
                self._wav_file.setsampwidth(2)
                self._wav_file.setnchannels(1)

            assert self._chunk_sample_rate is not None
            if self._chunk_sample_rate != _SAMPLE_RATE:
                input_chunk = np.frombuffer(chunk.audio, dtype=np.int16)
                num_new_samples = int(len(input_chunk) * _SAMPLE_RATE / self._chunk_sample_rate)
                resampled_chunk = resample(input_chunk, num_new_samples).astype(np.int16)
                self._wav_file.writeframes(resampled_chunk)
            else:
                self._wav_file.writeframes(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped. Transcribing %s", self._wav_path)
            assert self._wav_file is not None

            self._wav_file.close()
            self._wav_file = None
            self._chunk_sample_rate = None

            if self._wav_debug_dir is not None:
                try:
                    # Ensure the debug directory exists
                    os.makedirs(self._wav_debug_dir, exist_ok=True)

                    # Timestamp suffix with microseconds to avoid collisions
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    base = os.path.splitext(os.path.basename(self._wav_path))[0]
                    dst_name = f"{base}_{timestamp}.wav"
                    dst_path = os.path.join(self._wav_debug_dir, dst_name)

                    # Use shutil.copy2 to preserve metadata
                    shutil.copy2(self._wav_path, dst_path)

                    _LOGGER.debug("WAV debug copy written to %s", dst_path)
                except Exception as exc:
                    _LOGGER.exception("Failed to write WAV debug copy: %s", exc)

            async with self.model_lock:
                text = self.model.transcribe(self._wav_path)

            _LOGGER.info("Transciption: %s", text)
            await self.write_event(Transcript(text=text).event())
            _LOGGER.debug("Completed request")

            # Reset
            self._language = self.cli_args.language

            return False

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                self._language = transcribe.language
                _LOGGER.debug("Language set to %s", transcribe.language)
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True
