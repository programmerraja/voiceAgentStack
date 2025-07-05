#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import time
from typing import Optional

import numpy as np
from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams

try:
    import webrtcvad as webrtcvads
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use WebRTC VAD, you need to `pip install webrtcvad`.")
    raise Exception(f"Missing module(s): {e}")


class WebRTCVADAnalyzer(VADAnalyzer):
    def __init__(self, *, sample_rate: Optional[int] = None, params: Optional[VADParams] = None, aggressiveness: int = 2):
        super().__init__(sample_rate=sample_rate, params=params)
        self.vad = webrtcvads.Vad(aggressiveness)
        logger.debug("Initialized WebRTC VAD with aggressiveness %d", aggressiveness)

    def set_sample_rate(self, sample_rate: int):
        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(
                f"WebRTC VAD sample rate needs to be 8000, 16000, 32000, or 48000 (sample rate: {sample_rate})"
            )
        super().set_sample_rate(sample_rate)

    def num_frames_required(self) -> int:
        # WebRTC VAD expects frames of 10, 20, or 30 ms. We'll use 30 ms for max compatibility.
        # samples = sample_rate * duration_ms / 1000
        return int(self.sample_rate * 0.03)

    def voice_confidence(self, buffer) -> float:
        try:
            # WebRTC VAD expects 16-bit mono PCM
            # If buffer is not bytes, convert it
            if isinstance(buffer, np.ndarray):
                audio_bytes = buffer.astype(np.int16).tobytes()
            else:
                audio_bytes = buffer
            is_speech = self.vad.is_speech(audio_bytes, self.sample_rate)
            return 1.0 if is_speech else 0.0
        except Exception as e:
            logger.error(f"Error analyzing audio with WebRTC VAD: {e}")
            return 0.0
