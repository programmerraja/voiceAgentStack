FROM python:3.11-slim

# Install dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg

# Install Python deps
RUN pip install faster-whisper websockets numpy

# Copy server
WORKDIR /app
COPY server.py .

EXPOSE 8000

CMD ["python3", "server.py"]
