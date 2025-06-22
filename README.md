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
