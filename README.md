# Voice Agent Stack

This project is a full-stack voice agent application that enables real-time, voice-to-voice conversations with an AI assistant. It uses the `pipecat-ai` framework to orchestrate the various components of the voice pipeline.

## Overview

The application consists of a web-based client and a Python backend. The client captures the user's voice from the microphone and streams it to the backend. The backend processes the audio using a Speech-to-Text (STT) engine, sends the transcribed text to a Large Language Model (LLM) to generate a response, synthesizes the response back into audio using a Text-to-Speech (TTS) engine, and streams the audio back to the client for playback.

## Features

- **Real-time Voice Communication**: Low-latency, full-duplex audio streaming.
- **Speech-to-Text**: Transcribes user's speech into text using Whisper.
- **AI-Powered Responses**: Generates conversational responses using OLLama.
- **Text-to-Speech**: Synthesizes text responses into natural-sounding speech using a custom Kokoro TTS engine.
- **Web-based Client**: Simple and intuitive browser-based interface for interacting with the voice agent.

## Technology Stack

### Backend

- **Python**
- **FastAPI**: High-performance web framework for building APIs.
- **pipecat-ai**: Framework for building real-time voice and multimodal AI applications.
- **Whisper**: Speech-to-Text engine.
- **OLLama**: Large Language Model service.
- **Kokoro TTS**: Custom Text-to-Speech engine.
- **Uvicorn**: ASGI server for running the FastAPI application.

### Frontend

- **TypeScript**
- **Vite**: Next-generation frontend tooling.
- **@pipecat-ai/client-js**: Client-side library for `pipecat-ai`.

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js and npm
- OLLama installed and running.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd voice-agent-stack
    ```

2.  **Download Kokoro TTS model and voices:**
    ```bash
    mkdir -p bot/assets
    wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v1.0.int8.onnx -O bot/assets/kokoro-v1.0.int8.onnx
    wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json -O bot/assets/voices.json
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install client dependencies:**
    ```bash
    cd client
    npm install
    ```

### Running the Application

1.  **Start the backend server:**
    From the root directory, run:
    ```bash
    python server.py
    ```
    The server will start on `http://localhost:7860`.

2.  **Start the client development server:**
    In a separate terminal, from the `client` directory, run:
    ```bash
    npm run dev
    ```
    The client will be available at `http://localhost:5173`.

3.  **Interact with the application:**
    Open your browser and navigate to `http://localhost:5173`. Click the "Connect" button to start a conversation with the voice agent.

## Project Structure

```
.
├── bot/
│   └── bot_websocket_server.py   # Core pipecat pipeline logic
├── client/
│   ├── src/
│   │   └── app.ts                # Frontend application logic
│   └── index.html                # Main HTML file
├── server.py                     # FastAPI server
├── service/
│   └── Kokoro/
│       └── tts.py                # Custom Kokoro TTS service
├── requirements.txt              # Python dependencies
└── ...
```




XTTS
- $ docker run -e COQUI_TOS_AGREED=1 --rm -p 8000:80 ghcr.io/coqui-ai/xtts-streaming-server:latest-cpu

Orpheus
- git@github.com:Lex-au/Orpheus-FastAPI.git

python -m http.server 8080


/path/to/uv/python -m cProfile -o profile.out your_script.py

def log_time_async(threshold=1.0):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            cls = self.__class__.__name__
            method = func.__name__
            start = time.time()
            result = await func(self, *args, **kwargs)
            elapsed = time.time() - start
            if elapsed > threshold:
                logger.warning(f"{cls}.{method} took {elapsed:.2f}s (> {threshold}s)")
            return result
        return wrapper
    return decorator



| Spec                        | Detail                                                     |
| --------------------------- | ---------------------------------------------------------- |
| **GPU Architecture**        | Turing (TU104GL)                                           |
| **CUDA Cores**              | 2,560                                                      |
| **Tensor Cores**            | 320                                                        |
| **RT Cores**                | None (no real-time ray tracing)                            |
| **Base Clock**              | \~585 MHz                                                  |
| **Boost Clock**             | \~1,590 MHz                                                |
| **Memory**                  | 16 GB GDDR6                                                |
| **Memory Bandwidth**        | \~320 GB/s                                                 |
| **TDP**                     | 70 Watts                                                   |
| **Interface**               | PCIe 3.0 x16                                               |
| **Form Factor**             | Low-profile, single-slot                                   |
| **FP32 (Single-Precision)** | \~8.1 TFLOPS                                               |
| **FP16 (Half-Precision)**   | \~65 TFLOPS (with Tensor Cores)                            |
| **INT8**                    | \~130 TOPS                                                 |
| **Target Use**              | AI inference, ML training (small scale), video transcoding |
| **Virtual Display Output**  | None (headless)                                            |


nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv