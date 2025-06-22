import asyncio
from queue import Queue
from threading import Thread
from typing import AsyncGenerator, List, Optional

from loguru import logger
from orpheus_tts import OrpheusModel
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService


class OrpheusTTSService(TTSService):
    """TTS service for Orpheus.

    This service uses Orpheus to generate speech. It streams the audio chunks.
    """

    class InputParams(BaseModel):
        """Configuration parameters for Orpheus TTS service."""

        voice: str = Field("tara", description="Voice to use for generation.")
        repetition_penalty: Optional[float] = Field(1.1)
        stop_token_ids: Optional[List[int]] = Field([128258])
        max_tokens: Optional[int] = Field(2000)
        temperature: Optional[float] = Field(0.4)
        top_p: Optional[float] = Field(0.9)

    def __init__(
        self,
        *,
        model_name: str = "canopylabs/orpheus-tts-0.1-finetune-prod",
        sample_rate: int = 24000,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        """Initialize Orpheus TTS service.

        Args:
            model_name: The name of the Orpheus model to use.
            sample_rate: The sample rate of the audio.
            params: Configuration parameters for TTS generation.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        logger.info(f"Initializing Orpheus TTS service with model: {model_name}")
        self._model = OrpheusModel(model_name=model_name)
        self._settings = params.dict()
        logger.info("Orpheus TTS service initialized")

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS for: [{text}]")
        try:
            await self.start_ttfb_metrics()
            yield TTSStartedFrame()

            loop = asyncio.get_running_loop()
            q = Queue()

            def generate():
                try:
                    stream = self._model.generate_speech(
                        prompt=text,
                        voice=self._settings["voice"],
                        repetition_penalty=self._settings["repetition_penalty"],
                        stop_token_ids=self._settings["stop_token_ids"],
                        max_tokens=self._settings["max_tokens"],
                        temperature=self._settings["temperature"],
                        top_p=self._settings["top_p"],
                    )
                    for chunk in stream:
                        q.put(chunk)
                except Exception as e:
                    logger.error(
                        f"Error in Orpheus generate_speech thread: {e}", exc_info=True
                    )
                    q.put(e)
                finally:
                    q.put(None)  # Sentinel to indicate end of stream

            thread = Thread(target=generate)
            thread.start()

            await self.start_tts_usage_metrics(text)

            while True:
                item = await loop.run_in_executor(None, q.get)
                if isinstance(item, Exception):
                    raise item
                if item is None:
                    break

                yield TTSAudioRawFrame(
                    audio=item, sample_rate=self.sample_rate, num_channels=1
                )

            thread.join()

            yield TTSStoppedFrame()
        except Exception as e:
            logger.error(f"{self} exception: {e}", exc_info=True)
            yield ErrorFrame(f"Error generating audio: {str(e)}")