# Copyright (c) 2024–2025, Daily
# SPDX-License-Identifier: BSD 2-Clause License

from typing import Optional
import numpy as np
from loguru import logger
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams

try:
    from ten_vad import TenVad
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use TEN VAD, you need to `pip install git+https://github.com/TEN-framework/ten-vad.git`.")
    # raise Exception(f"Missing module(s): {e}")

class TENVADAnalyzer(VADAnalyzer):
    def __init__(self, *, sample_rate: Optional[int] = None, params: Optional[VADParams] = None, threshold: float = 0.5):
        super().__init__(sample_rate=sample_rate, params=params)
        self.threshold = threshold
        self.vad = TenVad(threshold=threshold)
        logger.debug(f"Initialized TEN VAD with threshold {threshold}")

    def set_sample_rate(self, sample_rate: int):
        if sample_rate != 16000:
            raise ValueError(
                f"TEN VAD sample rate needs to be 16000 (sample rate: {sample_rate})"
            )
        super().set_sample_rate(sample_rate)

    def num_frames_required(self) -> int:
        # TEN VAD is optimized for 10ms or 16ms frames at 16kHz
        # We'll use 16ms (256 samples) for max compatibility
        return 256

    def voice_confidence(self, buffer) -> float:
        try:
            # TEN VAD expects 16-bit mono PCM at 16kHz
            if isinstance(buffer, np.ndarray):
                audio_bytes = buffer.astype(np.int16).tobytes()
            else:
                audio_bytes = buffer
            # The TenVad class returns 1 for speech, 0 for non-speech
            is_speech = self.vad.is_speech(audio_bytes, self.sample_rate)
            return 1.0 if is_speech else 0.0
        except Exception as e:
            logger.error(f"Error analyzing audio with TEN VAD: {e}")
            return 0.0
