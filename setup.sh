mkdir -p /app/bot/assets
# wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx -O /app/bot/assets/kokoro-v1.0.int8.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16-gpu.onnx -O /app/bot/assets/kokoro-v1.0.fp16-gpu.onnx
wget https://huggingface.co/NeuML/kokoro-base-onnx/resolve/main/voices.json -O /app/bot/assets/voices.json
curl -fsSL https://ollama.com/install.sh | sh 
# ollama serve &
# ollama pull smollm &
docker run --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:latest
