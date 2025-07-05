FROM python:3.11-slim

# Create user with ID 1000 (required for HF Spaces)
RUN useradd -m -u 1000 user

# Install dependencies as root
RUN apt update && apt install -y wget curl

# Set up working directory and change ownership
WORKDIR /app
RUN chown -R user:user /app

# Install Python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    which uvicorn && uvicorn --version

# Copy code and setup with proper ownership
COPY --chown=user:user . .
RUN chmod +x ./setup.sh && ./setup.sh

# Switch to user before creating directories
USER user

# Set up Ollama directories with proper permissions
ENV HOME=/home/user
ENV OLLAMA_MODELS=/home/user/.ollama
RUN mkdir -p /home/user/.ollama

# Set PATH for user's local bin
ENV PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

docker run --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:latest

docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:v0.2.4


CMD bash -c "ollama serve & sleep 5 && ollama pull smollm && uvicorn server:app --host 0.0.0.0 --port 7860"


# GPU-enabled deployment
docker run --gpus=all --publish 8002:8000 \
  --volume ~/.cache/huggingface:/root/.cache/huggingface \
  fedirz/faster-whisper-server:latest-cuda

# CPU-only deployment
docker run --publish 8000:8000 \
  --volume ~/.cache/huggingface:/root/.cache/huggingface \
  fedirz/faster-whisper-server:latest-cpu
