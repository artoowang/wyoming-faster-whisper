"""Event handler for clients of the server."""
import argparse
import asyncio
import json
import logging
import os
import tempfile
import wave
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import sentencepiece
import sphn
from huggingface_hub import hf_hub_download
from moshi_mlx import models, utils

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
        stt_config = lm_config.get("stt_config", None)

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


    def transcribe(self, audio_path: str) -> str:
        audio, _ = sphn.read(audio_path, sample_rate=_SAMPLE_RATE)
        assert audio.ndim == 2 and audio.shape[0] == 1, "Expected mono audio"
        return "Test transcription"


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


    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            if self._wav_file is None:
                assert (chunk.rate == _SAMPLE_RATE,
                        f"Only supports {_SAMPLE_RATE} Hz audio, but received {chunk.rate}")
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)

            self._wav_file.writeframes(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug(
                "Audio stopped. Transcribing with initial prompt=%s",
                self.initial_prompt,
            )
            assert self._wav_file is not None

            self._wav_file.close()
            self._wav_file = None

            async with self.model_lock:
                text = self.model.transcribe(self._wav_path)

            _LOGGER.info(text)
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
