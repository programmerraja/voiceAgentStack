import asyncio
from typing import AsyncGenerator

import numpy as np
import torch
from dia.model import Dia
from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService


class DiaTTSService(TTSService):
    """TTS service for Dia.
    This service uses Dia to generate speech.
    It does not support streaming and will generate the entire audio at once.
    """

    class InputParams(BaseModel):
        """Configuration parameters for Dia TTS service."""

        use_torch_compile: bool = Field(False)
        verbose: bool = Field(False)

    def __init__(
        self,
        *,
        model_name: str = "nari-labs/Dia-1.6B",
        compute_dtype: str = "float32",
        device: str = "cpu",
        sample_rate: int = 24000,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        """Initialize Dia TTS service."""
        super().__init__(sample_rate=sample_rate, **kwargs)
        logger.info(f"Initializing Dia TTS service with model: {model_name}")

        torch_device = torch.device(device)

        self._model = Dia.from_pretrained(
            model_name, compute_dtype=compute_dtype, device=torch_device
        )
        self._settings = params.dict()
        logger.info("Dia TTS service initialized")

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS for: [{text}]")
        try:
            await self.start_ttfb_metrics()
            yield TTSStartedFrame()

            loop = asyncio.get_running_loop()

            await self.start_tts_usage_metrics(text)

            output = await loop.run_in_executor(
                None,
                self._model.generate,
                text,
                self._settings["use_torch_compile"],
                self._settings["verbose"],
            )

            audio_tensor = output["audio_tensor"]

            # The tensor is float32 in range [-1, 1], shape (1, N).
            # Convert to int16 bytes for pipecat.
            audio_data = (audio_tensor.cpu().numpy() * 32767).astype(np.int16).tobytes()

            yield TTSAudioRawFrame(
                audio=audio_data,
                sample_rate=self.sample_rate,
                num_channels=1,
            )

            yield TTSStoppedFrame()
        except Exception as e:
            logger.error(f"{self} exception: {e}", exc_info=True)
            yield ErrorFrame(f"Error generating audio: {str(e)}")