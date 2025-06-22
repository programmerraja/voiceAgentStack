import asyncio
import base64
import re
import tempfile
from typing import AsyncGenerator, Optional, List

import aiohttp
import numpy as np
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
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
from pipecat.transcriptions.language import Language


class ChatterboxTTSService(TTSService):
    """Text-to-Speech service using Chatterbox for on-device TTS.

    This service uses Chatterbox to generate speech. It supports voice cloning
    from an audio prompt.
    """

    class InputParams(BaseModel):
        """Configuration parameters for Chatterbox TTS service."""

        audio_prompt: Optional[str] = Field(
            None, description="URL or file path to an audio prompt for voice cloning."
        )
        exaggeration: float = Field(0.5, ge=0.0, le=1.0)
        cfg: float = Field(0.5, ge=0.0, le=1.0)
        temperature: float = Field(0.8, ge=0.0, le=1.0)

    def __init__(
        self,
        *,
        device: str = "cpu",
        params: InputParams = InputParams(),
        **kwargs,
    ):
        """Initialize Chatterbox TTS service.

        Args:
            device: The device to run the model on (e.g., "cpu", "cuda").
            params: Configuration parameters for TTS generation.
        """
        super().__init__(**kwargs)
        logger.info(f"Initializing Chatterbox TTS service on device: {device}")
        self._model = ChatterboxTTS.from_pretrained(device=device)
        self._sample_rate = self._model.sr
        self._settings = params.dict()
        self._temp_files: List[str] = []
        logger.info("Chatterbox TTS service initialized")

    def __del__(self):
        self._cleanup_temp_files()

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Returns the language code for Chatterbox TTS. Only English is supported."""
        if language.value.startswith("en"):
            return "en"
        logger.warning(
            f"Chatterbox TTS only supports English, but got {language}. Defaulting to English."
        )
        return "en"

    async def _handle_audio_prompt(self, audio_prompt: str) -> Optional[str]:
        if re.match(r"^https?://", audio_prompt):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(audio_prompt) as resp:
                        resp.raise_for_status()
                        content = await resp.read()
                        tmp_file = tempfile.NamedTemporaryFile(
                            delete=False, suffix=".wav"
                        )
                        tmp_file.write(content)
                        tmp_file.close()
                        self._temp_files.append(tmp_file.name)
                        return tmp_file.name
            except Exception as e:
                logger.error(f"Error downloading audio prompt from URL: {e}")
                return None
        return audio_prompt

    def _cleanup_temp_files(self):
        import os

        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except OSError as e:
                logger.warning(f"Error cleaning up temp file {temp_file}: {e}")
        self._temp_files.clear()

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Chatterbox."""
        logger.debug(f"Generating TTS for: [{text}]")

        try:
            await self.start_ttfb_metrics()
            yield TTSStartedFrame()

            audio_prompt_path = self._settings.get("audio_prompt")
            if audio_prompt_path:
                audio_prompt_path = await self._handle_audio_prompt(audio_prompt_path)

            await self.start_tts_usage_metrics(text)
            
            loop = asyncio.get_running_loop()
            wav = await loop.run_in_executor(
                None,
                self._model.generate,
                text,
                audio_prompt_path,
                self._settings["exaggeration"],
                self._settings["cfg"],
                self._settings["temperature"],
            )

            audio_data = (wav.cpu().numpy() * 32767).astype(np.int16).tobytes()
            yield TTSAudioRawFrame(
                audio=audio_data,
                sample_rate=self._sample_rate,
                num_channels=1,
            )

            yield TTSStoppedFrame()
        except Exception as e:
            logger.error(f"{self} exception: {e}", exc_info=True)
            yield ErrorFrame(f"Error generating audio: {e}")
        finally:
            self._cleanup_temp_files()

