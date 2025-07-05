FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install Python & system dependencies
RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        python3-dev \
        ffmpeg \
        git \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    websockets \
    numpy \
    aiohttp \
    chatterbox-tts

# Create working directory
WORKDIR /app

# Copy server file
COPY server.py .

# Expose port
EXPOSE 9801

# Set environment variables for CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Run the server
CMD ["python3", "server.py"] 