FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        ffmpeg \
        git \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
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

# Run the server
CMD ["python", "server.py"] 