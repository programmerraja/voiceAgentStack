#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware

import traceback
import faulthandler

faulthandler.enable()

# Load environment variables
load_dotenv(override=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles FastAPI startup and shutdown."""
    yield  # Run app


# Initialize FastAPI app with lifespan manager
app = FastAPI(lifespan=lifespan)

# Configure CORS to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok"}  # 🔁 HTTP 200


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted")
    try:
        # Serve Moshi streaming STT over /ws in an OpenAI-compatible schema
        from bot.voice_agent import VoiceAgent

        bot = VoiceAgent(websocket)
        await bot.run()
    except Exception as e:
        #print stack trace and line number
        print(f"Exception in run_bot: {e}",e)
        print(traceback.print_exc())


@app.post("/connect")
async def bot_connect(request: Request) -> Dict[Any, Any]:
    # return {"ws_url": "ws://localhost:7860/ws"}
    ws_url = f"wss://{request.headers.get('Host')}/ws"
    return {"ws_url": ws_url}


async def main():
    config = uvicorn.Config(app, host="0.0.0.0", port=7860)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    import signal

    async def serve():
        config = uvicorn.Config(app, host="0.0.0.0", port=7860)
        server = uvicorn.Server(config)
        await server.serve()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(serve())
    except KeyboardInterrupt:
        print("Received exit signal (Ctrl+C), shutting down...")
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
