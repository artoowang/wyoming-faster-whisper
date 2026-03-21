#!/usr/bin/env python3
import argparse
import asyncio
import logging
import platform
import re
from functools import partial

import faster_whisper
import mlx.core as mx
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .faster_whisper_event_handler import FasterWhisperEventHandler

_LOGGER = logging.getLogger(__name__)


def get_whisper_model_name(model: str) -> str:
    """Converts a user-friendly Whisper model to actual HuggingFace model name.

    This is remained from the original code, but TBH I do not fully understand
    why this is needed. In all of my testing, I do not use int8 models, so
    this code path is likely never used in the past, and we just use the
    provided model name (e.g., "large") directly.
    """
    match = re.match(r"^(tiny|base|small|medium)[.-]int8$", model)
    if match:
        model_size = match.group(1)
        model_name = f"{model_size}-int8"
        return f"rhasspy/faster-whisper-{model_name}"
    return model


def get_whisper_beam_size(beam_size: int) -> int:
    """Determines the beam size to use for Whisper model.

    If `beam_size` is greater than 0, returns it directly. Otherwise, it is
    determined based on the machine.
    """
    if beam_size > 0:
        return beam_size
    machine = platform.machine().lower()
    is_arm = ("arm" in machine) or ("aarch" in machine)
    beam_size = 1 if is_arm else 5
    _LOGGER.debug("Beam size automatically selected: %s", beam_size)
    return beam_size


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Name of the model to use.",
    )
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--download-dir",
        help="Directory to download models into",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for inference (default: cpu). Only used for faster-whisper model type.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Default language to set for transcription, or None for auto detection. Only used by transformer model type.",
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
        help="Model type: kyutai-stt (default), faster-whisper, transformer, whisper-mps, or glm-asr",
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
        help=(
            "Path to .rnnn model file for ffmpeg arnndn denoising. "
            + "Required if --ffmpeg-denoise is set."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    server = AsyncServer.from_uri(args.uri)
    model_lock = asyncio.Lock()

    if args.model_type == "transformer":
        args.model = get_whisper_model_name(args.model)
        args.beam_size = get_whisper_beam_size(args.beam_size)

        _LOGGER.debug("Loading %s", args.model)
        from .transformers_whisper import (
            TransformersWhisperEventHandler,
            TransformersWhisperModel,
        )

        assert args.download_dir
        transformer_model = TransformersWhisperModel(
            args.model, args.download_dir, args.local_files_only
        )
        wyoming_info = Info(
            asr=[
                AsrProgram(
                    name="transformers-whisper",
                    description="HuggingFace Transformers Whisper",
                    attribution=Attribution(
                        name="HuggingFace",
                        url="https://huggingface.co/",
                    ),
                    installed=True,
                    version=__version__,
                    models=[
                        AsrModel(
                            name=args.model,
                            description=args.model,
                            attribution=Attribution(
                                name="HuggingFace",
                                url="https://huggingface.co/",
                            ),
                            installed=True,
                            languages=[],
                            version="",
                        )
                    ],
                )
            ],
        )
        _LOGGER.info("Ready")

        await server.run(
            partial(
                TransformersWhisperEventHandler,
                wyoming_info,
                args.language,
                args.beam_size,
                transformer_model,
                model_lock,
            )
        )
    elif args.model_type == "faster-whisper":
        args.model = get_whisper_model_name(args.model)
        args.beam_size = get_whisper_beam_size(args.beam_size)

        _LOGGER.debug("Loading %s", args.model)
        assert args.download_dir
        faster_whisper_model = faster_whisper.WhisperModel(
            args.model,
            download_root=args.download_dir,
            device=args.device,
            compute_type=args.compute_type,
        )

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
                            name=args.model,
                            description=args.model,
                            attribution=Attribution(
                                name="Systran",
                                url="https://huggingface.co/Systran",
                            ),
                            installed=True,
                            languages=[],
                            version=faster_whisper.__version__,
                        )
                    ],
                )
            ],
        )
        _LOGGER.info("Ready")

        await server.run(
            partial(
                FasterWhisperEventHandler,
                wyoming_info,
                args,
                faster_whisper_model,
                model_lock,
                initial_prompt=args.initial_prompt,
            )
        )
    elif args.model_type == "kyutai-stt":
        from .kyutai_stt_handler import KyutaiSttEventHandler, KyutaiSttModel

        _LOGGER.debug("Loading %s", args.model)
        kyutai_model = KyutaiSttModel(hf_repo=args.model)
        wyoming_info = Info(
            asr=[
                AsrProgram(
                    name="kyutai-stt",
                    description="Kyutai's STT model (MLX-based, Apple Silicon optimized)",
                    attribution=Attribution(
                        name="Kyutai",
                        url="https://kyutai.org/",
                    ),
                    installed=True,
                    version=__version__,
                    models=[
                        AsrModel(
                            name=args.model,
                            description=args.model,
                            attribution=Attribution(
                                name="Kyutai",
                                url="https://huggingface.co/kyutai",
                            ),
                            installed=True,
                            languages=[],
                            version="",
                        )
                    ],
                )
            ],
        )
        _LOGGER.info("Ready")

        await server.run(
            partial(
                KyutaiSttEventHandler,
                wyoming_info,
                args,
                kyutai_model,
                model_lock,
                initial_prompt=args.initial_prompt,
            )
        )
    elif args.model_type == "whisper-mps":
        from whisper_mps.whisper.transcribe import ModelHolder
        from .whisper_mps_event_handler import WhisperMpsEventHandler

        args.model = get_whisper_model_name(args.model)

        _LOGGER.debug("Loading %s", args.model)
        ModelHolder.get_model(args.model, mx.float16)
        wyoming_info = Info(
            asr=[
                AsrProgram(
                    name="whisper-mps",
                    description="whisper-mps for Apple Silicon MPS acceleration",
                    attribution=Attribution(
                        name="MJ vGruter",
                        url="https://github.com/Vaesen011/whisper-mps",
                    ),
                    installed=True,
                    version=__version__,
                    models=[
                        AsrModel(
                            name=args.model,
                            description=args.model,
                            attribution=Attribution(
                                name="MJ vGruter",
                                url="https://github.com/Vaesen011/whisper-mps",
                            ),
                            installed=True,
                            languages=[],
                            version="",
                        )
                    ],
                )
            ],
        )
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
    elif args.model_type == "glm-asr":
        from .glm_asr_event_handler import GlmAsrEventHandler, GlmAsrModel

        _LOGGER.debug("Loading %s", args.model)
        glm_model = GlmAsrModel(
            args.model,
            cache_dir=args.download_dir,
            local_files_only=args.local_files_only,
        )
        wyoming_info = Info(
            asr=[
                AsrProgram(
                    name="glm-asr",
                    description="GLM-ASR speech recognition model",
                    attribution=Attribution(
                        name="THUDM",
                        url="https://github.com/THUDM/GLM-ASR",
                    ),
                    installed=True,
                    version=__version__,
                    models=[
                        AsrModel(
                            name=args.model,
                            description=args.model,
                            attribution=Attribution(
                                name="THUDM",
                                url="https://huggingface.co/zai-org",
                            ),
                            installed=True,
                            languages=[],
                            version="",
                        )
                    ],
                )
            ],
        )
        _LOGGER.info("Ready")

        await server.run(
            partial(
                GlmAsrEventHandler,
                wyoming_info,
                glm_model,
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
