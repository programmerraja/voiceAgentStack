# new file
import asyncio
import base64
import json
import os
import time
import uuid
from typing import List, Optional

import numpy as np
import torch
from fastapi import WebSocket, FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

from moshi.models import loaders, MimiModel, LMModel, LMGen
import sentencepiece


class InferenceState:
    """Holds the heavy models (MIMI encoder + LM + tokenizer) and streaming generators"""

    def __init__(
        self,
        mimi: MimiModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm: LMModel,
        batch_size: int,
        device: str | torch.device,
    ):
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        self.lm_gen = LMGen(lm, temp=0, temp_text=0, use_sampling=False)
        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.batch_size = batch_size
        # prepare streaming forever (warm-up internal state)
        self.mimi.streaming_forever(batch_size)
        self.lm_gen.streaming_forever(batch_size)


class StreamingTranscriber:
    """Stream PCM samples into InferenceState and yield text deltas"""

    def __init__(self, state: InferenceState):
        self.state = state
        self.residual: np.ndarray = np.empty(0, dtype=np.float32)
        self.first_frame = True
        self._all_text: List[str] = []

    def feed(self, samples: np.ndarray) -> List[str]:
        """Feed float32 mono samples in range [-1,1] and return list of new text deltas."""
        if samples.ndim != 1:
            samples = samples.flatten()
        self.residual = np.concatenate([self.residual, samples])
        new_text: List[str] = []
        frame = self.state.frame_size
        while self.residual.shape[0] >= frame:
            chunk = self.residual[:frame].copy()
            self.residual = self.residual[frame:]
            chunk_tensor = torch.from_numpy(chunk).to(self.state.device).unsqueeze(0).unsqueeze(0)
            codes = self.state.mimi.encode(chunk_tensor)
            # For the very first frame, call step twice to prime LM as in original code
            if self.first_frame:
                _ = self.state.lm_gen.step(codes)
                self.first_frame = False
            tokens = self.state.lm_gen.step(codes)
            if tokens is None:
                continue
            token_id = tokens[0, 0].cpu().item()
            # Skip special tokens 0 and 3
            if token_id not in (0, 3):
                text_piece = self.state.text_tokenizer.id_to_piece(token_id).replace("▁", " ")
                new_text.append(text_piece)
                self._all_text.append(text_piece)
        return new_text

    @property
    def transcript(self) -> str:
        return "".join(self._all_text)


# -------------------- Model lazy loader --------------------

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_REPO = os.getenv("MOSHI_MODEL_REPO", "kyutai/stt-1b-en_fr")

# Global singleton heavy models, loaded lazily the first time we need them.
_MIMI: Optional[MimiModel] = None
_LM: Optional[LMModel] = None
_TOKENIZER: Optional[sentencepiece.SentencePieceProcessor] = None


def _load_models():
    global _MIMI, _LM, _TOKENIZER
    if _MIMI is None:
        ckpt = loaders.CheckpointInfo.from_hf_repo(_MODEL_REPO)
        _MIMI = ckpt.get_mimi(device=_DEVICE)
        _LM = ckpt.get_moshi(device=_DEVICE)
        _TOKENIZER = ckpt.get_text_tokenizer()


# -------------------- WebSocket handler --------------------

async def run_moshi_stt(websocket: WebSocket):
    """Handle a single websocket connection implementing an OpenAI-like STT WS interface."""
    await websocket.accept()
    _load_models()
    assert _MIMI is not None and _LM is not None and _TOKENIZER is not None

    # Build per-connection inference state
    state = InferenceState(
        mimi=_MIMI,
        text_tokenizer=_TOKENIZER,
        lm=_LM,
        batch_size=1,
        device=_DEVICE,
    )
    transcriber = StreamingTranscriber(state)

    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    created_payload = {
        "type": "transcription_session.created",
        "session": {
            "id": session_id,
            "input_audio_format": "pcm24",
            "expires_at": int(time.time()) + 60 * 60,
        },
    }
    await websocket.send_json(created_payload)

    try:
        while True:
            msg = await websocket.receive()
            if "text" in msg:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                msg_type = data.get("type")
                if msg_type == "transcription_session.update":
                    # Echo back update confirmation
                    updated = {
                        "type": "transcription_session.updated",
                        "session": data.get("session", {}),
                    }
                    await websocket.send_json(updated)
                elif msg_type == "input_audio_buffer.append":
                    audio_b64 = data.get("audio")
                    if not audio_b64:
                        continue
                    raw_bytes = base64.b64decode(audio_b64)
                    pcm = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    deltas = transcriber.feed(pcm)
                    for delta in deltas:
                        payload = {
                            "type": "conversation.item.input_audio_transcription.delta",
                            "delta": delta,
                        }
                        await websocket.send_json(payload)
                elif msg_type == "input_audio_buffer.commit":
                    complete = {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "transcript": transcriber.transcript,
                    }
                    await websocket.send_json(complete)
                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                else:
                    # unsupported
                    pass
            elif "bytes" in msg and msg["bytes"] is not None:
                raw_bytes = msg["bytes"]
                pcm = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                deltas = transcriber.feed(pcm)
                for delta in deltas:
                    payload = {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "delta": delta,
                    }
                    await websocket.send_json(payload)
            elif msg.get("type") == "websocket.disconnect":
                break
    except Exception as e:
        # Send error payload to client
        await websocket.send_json({"type": "error", "error": str(e)})
    finally:
        await websocket.close()


app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await run_moshi_stt(websocket)

if __name__ == "__main__":
    uvicorn.run("service.moshi.ws_server:app", host="0.0.0.0", port=8000, reload=True) 