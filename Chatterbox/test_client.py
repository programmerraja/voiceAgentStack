#!/usr/bin/env python3
"""
Test client for Chatterbox TTS WebSocket server.
"""

import asyncio
import websockets
import wave
import time
import argparse
import sys
from pathlib import Path


async def test_tts_basic(uri: str, text: str, output_file: str):
    """Test basic TTS functionality"""
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected! Sending text: '{text}'")
            
            start_time = time.perf_counter()
            await websocket.send(text)
            
            # Receive audio data
            audio_data = await websocket.recv()
            processing_time = time.perf_counter() - start_time
            
            if isinstance(audio_data, bytes) and not audio_data.startswith(b"ERROR"):
                print(f"✓ Received {len(audio_data)} bytes of audio data")
                print(f"✓ Processing time: {processing_time:.3f} seconds")
                
                # Save as raw PCM file
                with open(output_file, "wb") as f:
                    f.write(audio_data)
                print(f"✓ Audio saved to {output_file}")
                
                # Also save as WAV file for easier playback
                wav_file = output_file.replace('.pcm', '.wav')
                save_as_wav(audio_data, wav_file)
                print(f"✓ Audio saved as WAV to {wav_file}")
                
            else:
                error_msg = audio_data.decode() if isinstance(audio_data, bytes) else str(audio_data)
                print(f"✗ Error: {error_msg}")
                return False
                
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False
    
    return True


def save_as_wav(audio_data: bytes, filename: str, sample_rate: int = 24000):
    """Save raw PCM data as WAV file"""
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)


async def test_multiple_requests(uri: str, texts: list, output_dir: str):
    """Test multiple TTS requests"""
    print(f"Testing multiple requests to {uri}...")
    
    Path(output_dir).mkdir(exist_ok=True)
    success_count = 0
    
    for i, text in enumerate(texts):
        output_file = f"{output_dir}/test_{i+1}.pcm"
        print(f"\nTest {i+1}/{len(texts)}: {text[:50]}...")
        
        if await test_tts_basic(uri, text, output_file):
            success_count += 1
        else:
            print(f"✗ Test {i+1} failed")
    
    print(f"\n✓ Completed {success_count}/{len(texts)} tests successfully")
    return success_count == len(texts)


async def test_concurrent_requests(uri: str, text: str, num_concurrent: int = 3):
    """Test concurrent TTS requests"""
    print(f"Testing {num_concurrent} concurrent requests...")
    
    async def single_request(request_id: int):
        try:
            async with websockets.connect(uri) as websocket:
                await websocket.send(f"{text} (Request {request_id})")
                audio_data = await websocket.recv()
                
                if isinstance(audio_data, bytes) and not audio_data.startswith(b"ERROR"):
                    print(f"✓ Request {request_id}: {len(audio_data)} bytes")
                    return True
                else:
                    print(f"✗ Request {request_id}: Error")
                    return False
        except Exception as e:
            print(f"✗ Request {request_id}: {e}")
            return False
    
    start_time = time.perf_counter()
    results = await asyncio.gather(*[single_request(i) for i in range(num_concurrent)])
    total_time = time.perf_counter() - start_time
    
    success_count = sum(results)
    print(f"✓ Concurrent test: {success_count}/{num_concurrent} succeeded in {total_time:.3f}s")
    
    return success_count == num_concurrent


def main():
    parser = argparse.ArgumentParser(description="Test Chatterbox TTS WebSocket server")
    parser.add_argument("--uri", default="ws://localhost:9801", help="WebSocket URI")
    parser.add_argument("--text", default="Hello, this is a test of Chatterbox TTS!", help="Text to synthesize")
    parser.add_argument("--output", default="test_output.pcm", help="Output file name")
    parser.add_argument("--test-type", choices=["basic", "multiple", "concurrent", "all"], 
                       default="basic", help="Type of test to run")
    parser.add_argument("--concurrent", type=int, default=3, help="Number of concurrent requests")
    
    args = parser.parse_args()
    
    test_texts = [
        "Hello, this is a test of Chatterbox TTS!",
        "The quick brown fox jumps over the lazy dog.",
        "This is a longer sentence to test the quality of the text-to-speech synthesis.",
        "Numbers: one, two, three, four, five. Years: 2024, 2025, 2026.",
        "Testing punctuation: Hello! How are you? I'm fine, thanks. What about you?",
    ]
    
    async def run_tests():
        if args.test_type in ["basic", "all"]:
            print("=== Basic TTS Test ===")
            success = await test_tts_basic(args.uri, args.text, args.output)
            if not success:
                return False
        
        if args.test_type in ["multiple", "all"]:
            print("\n=== Multiple Requests Test ===")
            success = await test_multiple_requests(args.uri, test_texts, "test_outputs")
            if not success:
                return False
        
        if args.test_type in ["concurrent", "all"]:
            print("\n=== Concurrent Requests Test ===")
            success = await test_concurrent_requests(args.uri, args.text, args.concurrent)
            if not success:
                return False
        
        return True
    
    try:
        success = asyncio.run(run_tests())
        if success:
            print("\n🎉 All tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 