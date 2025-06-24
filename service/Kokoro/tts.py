#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
# This code originally written by Marmik Pandya (marmikcfc - github.com/marmikcfc)


import numpy as np
import asyncio
from typing import AsyncGenerator, List, Optional, Union
from pipecat.utils.tracing.service_decorators import traced_tts


from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.services.tts_service import InterruptibleTTSService


# load Kokoro from kokoro-onnx
try:
    from kokoro_onnx import Kokoro
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Kokoro, you need to `pip install kokoro-onnx`. Also, download the model files from the Kokoro repository."
    )
    raise Exception(f"Missing module: {e}")


def language_to_kokoro_language(language: Language) -> Optional[str]:
    """Convert pipecat Language to Kokoro language code."""
    BASE_LANGUAGES = {
        Language.EN: "en-us",
        Language.FR: "fr-fr",
        Language.IT: "it",
        Language.JA: "ja",
        Language.CMN: "cmn"
        # Add more language mappings as supported by Kokoro
    }

    result = BASE_LANGUAGES.get(language)

    # If not found in base languages, try to find the base language from a variant
    if not result:
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        # Look up the base code in our supported languages
        result = f"{base_code}-us" if base_code in ["en"] else None

    return result


class KokoroTTSService(InterruptibleTTSService):
    """Text-to-Speech service using Kokoro for on-device TTS.
    
    This service uses Kokoro to generate speech without requiring external API connections.
    """
    class InputParams(BaseModel):
        """Configuration parameters for Kokoro TTS service."""
        language: Optional[Language] = Language.EN
        speed: Optional[float] = 1.0

    def __init__(
        self,
        *,
        model_path: str,
        voices_path: str,
        voice_id: str = "af_sarah",
        sample_rate: Optional[int] = None,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        """Initialize Kokoro TTS service.
        
        Args:
            model_path: Path to the Kokoro ONNX model file
            voices_path: Path to the Kokoro voices file
            voice_id: ID of the voice to use
            sample_rate: Output audio sample rate
            params: Additional configuration parameters
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        logger.info(f"Initializing Kokoro TTS service with model_path: {model_path} and voices_path: {voices_path}")
        self._kokoro = Kokoro(model_path, voices_path)
        logger.info(f"Kokoro initialized")
        self._settings = {
            "language": self.language_to_service_language(params.language)
            if params.language
            else "en-us",
            "speed": params.speed,
        }
        self.set_voice(voice_id)  # Presumably this sets self._voice_id
        
        # Initialize interrupt handling
        self._interrupt_event = asyncio.Event()
        self._current_stream = None

        logger.info("Kokoro TTS service initialized")

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert pipecat language to Kokoro language code."""
        return language_to_kokoro_language(language)

   
    async def _disconnect(self):
        """Handle disconnection and interrupt current streaming."""
        logger.info("Disconnecting Kokoro TTS service - stopping current stream")
        
        # Signal interruption
        if not self._interrupt_event.is_set():
            self._interrupt_event.set()
        
        # Reset the current stream reference
        self._current_stream = None
        
        logger.info("Kokoro TTS service disconnected")

    async def interrupt(self):
        """Public method to interrupt current TTS generation."""
        logger.info("Interrupting current TTS generation")
        await self._disconnect()
        
    async def _connect(self):
        """Handle connection - reset interrupt state."""
        logger.info("Connecting Kokoro TTS service")
        self._interrupt_event.clear()
    
    async def _disconnect_websocket(self):
        pass

    async def _connect_websocket(self):
        pass
    
    
    async def _receive_messages(self):
        pass
    
    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Kokoro in a streaming fashion.
        
        Args:
            text: The text to convert to speech
            
        Yields:
            Frames containing audio data and status information.
        """
        logger.debug(f"Generating TTS: [{text}]")
        
        # Clear any previous interrupt state
        self._interrupt_event.clear()
        
        try:
            await self.start_ttfb_metrics()
            yield TTSStartedFrame()

            # Use Kokoro's streaming mode. The create_stream method is assumed to return
            # an async generator that yields (samples, sample_rate) tuples, where samples is a numpy array.
            logger.info(f"Creating stream")
            self._current_stream = self._kokoro.create_stream(
                text,
                voice=self._voice_id,
                speed=self._settings["speed"],
                lang=self._settings["language"],
            )

            await self.start_tts_usage_metrics(text)
            started = False
            
            # Stream with interrupt checking - no need for asyncio.create_task
            async for samples, sample_rate in self._current_stream:
                # Check for interruption before processing each chunk
                if self._interrupt_event.is_set():
                    logger.info("TTS generation interrupted")
                    break
                    
                if not started:
                    started = True
                    
                # Convert the float32 samples (assumed in the range [-1, 1]) to int16 PCM format
                samples_int16 = (samples * 32767).astype(np.int16)
                yield TTSAudioRawFrame(
                    audio=samples_int16.tobytes(),
                    sample_rate=sample_rate,
                    num_channels=1,
                )

            yield TTSStoppedFrame()

        except asyncio.CancelledError:
            logger.info("TTS generation was cancelled")
            yield TTSStoppedFrame()
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(f"Error generating audio: {str(e)}")
        finally:
            # Clean up
            self._current_stream = None