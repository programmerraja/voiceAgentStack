# Chatterbox TTS WebSocket Server

A high-performance WebSocket-based Text-to-Speech server using Chatterbox for on-device TTS generation.

## Features

- **WebSocket Interface**: Real-time text-to-speech conversion
- **Voice Cloning**: Support for audio prompt-based voice cloning
- **Optimized Performance**: Automatic device selection (CPU/GPU)
- **Model Pre-warming**: Faster first inference
- **Configurable Parameters**: Adjustable exaggeration, CFG, and temperature
- **Streaming**: Efficient audio streaming via WebSocket

## Quick Start

### Using Docker (Recommended)

#### CPU Version
```bash
docker build -f docker-cpu.dockerfile -t chatterbox-cpu .
docker run -p 9801:9801 chatterbox-cpu
```

#### GPU Version
```bash
docker build -f docker-gpu.dockerfile -t chatterbox-gpu .
docker run --gpus all -p 9801:9801 chatterbox-gpu
```

### Manual Installation

1. Install dependencies:
```bash
pip install torch torchvision torchaudio websockets numpy aiohttp chatterbox-tts
```

2. Run the server:
```bash
python server.py
```

## Usage

The server listens on `ws://localhost:9801` by default.

### Basic Text-to-Speech

Send text messages to the WebSocket endpoint and receive audio data:

```python
import asyncio
import websockets

async def test_tts():
    uri = "ws://localhost:9801"
    async with websockets.connect(uri) as websocket:
        # Send text to convert to speech
        await websocket.send("Hello, this is a test of Chatterbox TTS!")
        
        # Receive audio data
        audio_data = await websocket.recv()
        
        # Save or play the audio
        with open("output.wav", "wb") as f:
            f.write(audio_data)

asyncio.run(test_tts())
```

### Advanced Configuration

The server supports various configuration parameters:

- `exaggeration`: Controls speech exaggeration (0.0-1.0, default: 0.5)
- `cfg`: Classifier-free guidance scale (0.0-1.0, default: 0.5)
- `temperature`: Controls randomness (0.0-1.0, default: 0.8)
- `audio_prompt`: Path or URL to audio file for voice cloning

## API Reference

### WebSocket Endpoint

- **URL**: `ws://localhost:9801`
- **Protocol**: WebSocket
- **Input**: Text string (UTF-8)
- **Output**: Audio data (PCM int16 bytes)

### Audio Format

- **Sample Rate**: Model dependent (typically 24000 Hz)
- **Channels**: 1 (mono)
- **Format**: PCM int16 (signed 16-bit)

## Performance Optimization

- **Device Selection**: Automatically uses GPU if available
- **Model Pre-warming**: Reduces cold start latency
- **Async Processing**: Non-blocking audio generation
- **Memory Management**: Automatic cleanup of temporary files

## Error Handling

The server sends error messages as bytes when issues occur:
- Model loading failures
- Invalid audio prompts
- Generation errors

## Development

### Testing

Use the included test client:
```bash
python test_client.py
```

### Monitoring

Server logs include:
- Connection events
- Processing times
- Error details
- Performance metrics

## License

This project follows the same license as the Chatterbox TTS library. 