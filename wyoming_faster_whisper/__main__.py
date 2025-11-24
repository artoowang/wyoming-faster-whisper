#!/usr/bin/env python3
import argparse
import asyncio
import logging
import mlx.core as mx
import platform
import re
from functools import partial
from typing import Any

import faster_whisper
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import FasterWhisperEventHandler

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Name of faster-whisper model to use (or auto)",
    )
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--download-dir",
        help="Directory to download models into",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for inference (default: cpu)",
    )
    parser.add_argument(
        "--language",
        help="Default language to set for transcription",
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        help="Compute type (float16, int8, etc.)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Size of beam during decoding (0 for auto)",
    )
    parser.add_argument(
        "--initial-prompt",
        help="Optional text to provide as a prompt for the first window",
    )
    parser.add_argument(
        "--model-type",
        default="kyutai-stt",
        help="Model type: kyutai-stt (default), faster-whisper, or transformer",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Don't check HuggingFace hub for updates every time",
    )
    #
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    parser.add_argument(
        "--audio-debug-dir",
        default=None,
        help="When supplied, dump the resampled audio file to this directory for debugging.",
    )
    parser.add_argument(
        "--ffmpeg-denoise",
        action="store_true",
        help="Use ffmpeg arnndn filter to denoise audio before transcription.",
    )
    parser.add_argument(
        "--ffmpeg-arnndn-model-path",
        default=None,
        help=("Path to .rnnn model file for ffmpeg arnndn denoising. " +
              "Required if --ffmpeg-denoise is set."),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    # Automatic configuration for ARM
    machine = platform.machine().lower()
    is_arm = ("arm" in machine) or ("aarch" in machine)
    if args.model == "auto":
        args.model = "tiny-int8" if is_arm else "base-int8"
        _LOGGER.debug("Model automatically selected: %s", args.model)

    if args.beam_size <= 0:
        args.beam_size = 1 if is_arm else 5
        _LOGGER.debug("Beam size automatically selected: %s", args.beam_size)

    # Resolve model name
    model_name = args.model
    match = re.match(r"^(tiny|base|small|medium)[.-]int8$", args.model)
    if match:
        # Original models re-uploaded to huggingface
        model_size = match.group(1)
        model_name = f"{model_size}-int8"
        args.model = f"rhasspy/faster-whisper-{model_name}"

    if args.language == "auto":
        # Whisper does not understand "auto"
        args.language = None

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="faster-whisper",
                description="Faster Whisper transcription with CTranslate2",
                attribution=Attribution(
                    name="Guillaume Klein",
                    url="https://github.com/guillaumekln/faster-whisper/",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=model_name,
                        description=model_name,
                        attribution=Attribution(
                            name="Systran",
                            url="https://huggingface.co/Systran",
                        ),
                        installed=True,
                        languages=faster_whisper.tokenizer._LANGUAGE_CODES,  # pylint: disable=protected-access
                        version=faster_whisper.__version__,
                    )
                ],
            )
        ],
    )

    server = AsyncServer.from_uri(args.uri)
    model_lock = asyncio.Lock()

    if args.model_type == "transformer":
        _LOGGER.debug("Loading %s", args.model)
        # Use HuggingFace transformers
        from .transformers_whisper import (
            TransformersWhisperEventHandler,
            TransformersWhisperModel,
        )
        assert args.download_dir
        model = TransformersWhisperModel(
            args.model, args.download_dir, args.local_files_only
        )
        _LOGGER.info("Ready")

        # TODO: initial prompt
        await server.run(
            partial(
                TransformersWhisperEventHandler,
                wyoming_info,
                args.language,
                args.beam_size,
                model,
                model_lock,
            )
        )
    elif args.model_type == "faster-whisper":
        _LOGGER.debug("Loading %s", args.model)
        # Use faster-whisper
        assert args.download_dir
        model = faster_whisper.WhisperModel(
            args.model,
            download_root=args.download_dir,
            device=args.device,
            compute_type=args.compute_type,
        )
        _LOGGER.info("Ready")

        await server.run(
            partial(
                FasterWhisperEventHandler,
                wyoming_info,
                args,
                model,
                model_lock,
                initial_prompt=args.initial_prompt,
            )
        )
    elif args.model_type == "kyutai-stt":
        # Use Kyutai STT
        from .kyutai_stt_handler import (
            KyutaiSttEventHandler,
            KyutaiSttModel,
        )
        _LOGGER.debug("Loading %s", args.model)
        model = KyutaiSttModel(hf_repo=args.model)
        _LOGGER.info("Ready")

        await server.run(
            partial(
                KyutaiSttEventHandler,
                wyoming_info,
                args,
                model,
                model_lock,
                initial_prompt=args.initial_prompt,
            )
        )
    elif args.model_type == "whisper-mps":
        # Use whisper-mps.
        from .whisper_mps_event_handler import (
            WhisperMpsEventHandler,
        )
        from whisper_mps.whisper.transcribe import ModelHolder

        _LOGGER.debug("Loading %s", args.model)
        # This preloads the model in ModelHolder, so later when whisper-mps
        # tries to load the same model name, it will reuse the already loaded
        # model.
        # TODO: Hard coded fp16 for now. We can use args.compute_type later.
        ModelHolder.get_model(args.model, mx.float16)
        _LOGGER.info("Ready")

        await server.run(
            partial(
                WhisperMpsEventHandler,
                wyoming_info,
                args,
                model_lock,
                initial_prompt=args.initial_prompt,
            )
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")


# -----------------------------------------------------------------------------


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
