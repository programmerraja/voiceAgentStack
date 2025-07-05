import asyncio
import websockets
import numpy as np
import tempfile
import os
import logging
import re
import time
from datetime import datetime, UTC
from typing import AsyncGenerator, Optional, List
import torch
import aiohttp
from chatterbox.tts import ChatterboxTTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration parameters
class ChatterboxConfig:
    def __init__(
        self,
        audio_prompt: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg: float = 0.5,
        temperature: float = 0.8,
    ):
        self.audio_prompt = audio_prompt
        self.exaggeration = max(0.0, min(1.0, exaggeration))
        self.cfg = max(0.0, min(1.0, cfg))
        self.temperature = max(0.0, min(1.0, temperature))


# Optimize device selection
def get_optimal_device():
    """Determine the best device for the system"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


device = get_optimal_device()
logger.info(f"Using device: {device}")

# Initialize Chatterbox model
logger.info("Loading Chatterbox model...")
model = ChatterboxTTS.from_pretrained(device=device)
sample_rate = model.sr
config = ChatterboxConfig()
temp_files: List[str] = []

logger.info(f"Chatterbox model loaded. Sample rate: {sample_rate}")


def cleanup_temp_files():
    """Clean up temporary files"""
    global temp_files
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except OSError as e:
            logger.warning(f"Error cleaning up temp file {temp_file}: {e}")
    temp_files.clear()


async def handle_audio_prompt(audio_prompt: str) -> Optional[str]:
    """Handle audio prompt URL or file path"""
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
                    temp_files.append(tmp_file.name)
                    return tmp_file.name
        except Exception as e:
            logger.error(f"Error downloading audio prompt from URL: {e}")
            return None
    return audio_prompt


async def run_tts(text: str, tts_config: ChatterboxConfig = None) -> AsyncGenerator[bytes, None]:
    """Generate speech from text using Chatterbox"""
    if not model:
        logger.error("Chatterbox model not available")
        yield b"Error: Chatterbox model not available"
        return

    if not tts_config:
        tts_config = config

    logger.debug(f"Generating TTS for: [{text}]")

    try:
        audio_prompt_path = tts_config.audio_prompt
        if audio_prompt_path:
            audio_prompt_path = await handle_audio_prompt(audio_prompt_path)

        # Run TTS generation in thread to avoid blocking
        loop = asyncio.get_running_loop()
        wav = await loop.run_in_executor(
            None,
            model.generate,
            text,
            audio_prompt_path,
            tts_config.exaggeration,
            tts_config.cfg,
            tts_config.temperature,
        )

        # Convert to int16 PCM bytes
        audio_data = (wav.cpu().numpy() * 32767).astype(np.int16).tobytes()
        yield audio_data

    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        yield f"ERROR: {str(e)}".encode()
    finally:
        cleanup_temp_files()


async def handler(websocket):
    logger.info("New WebSocket connection established")

    try:
        async for message in websocket:
            start_time = time.perf_counter()

            try:
                # Parse message as text (could be JSON for advanced config)
                text = message.decode('utf-8') if isinstance(message, bytes) else message
                
                # For now, use default config. Could extend to parse JSON for custom config
                tts_config = config
                
                # Generate and send audio
                async for audio_chunk in run_tts(text, tts_config):
                    await websocket.send(audio_chunk)

                processing_time = time.perf_counter() - start_time
                logger.info(f"Total processing time: {processing_time:.3f} seconds")

            except Exception as e:
                logger.error(f"Handler error: {e}")
                await websocket.send(f"ERROR: {str(e)}".encode())

    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")
    finally:
        cleanup_temp_files()


async def main():
    # Pre-warm the model with dummy text for faster first generation
    logger.info("Pre-warming model...")
    try:
        dummy_text = "Hello, this is a test."
        async for _ in run_tts(dummy_text):
            pass
        logger.info("Model pre-warmed successfully")
    except Exception as e:
        logger.warning(f"Model pre-warming failed: {e}")

    async with websockets.serve(handler, "0.0.0.0", 9801):
        logger.info("Chatterbox WebSocket server listening on ws://0.0.0.0:9801")
        logger.info(f"Model device: {device}, Sample rate: {sample_rate}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
