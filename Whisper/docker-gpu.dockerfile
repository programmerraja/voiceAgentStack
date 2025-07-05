FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install Python & common tools
RUN apt-get update && \
    apt-get install -y python3 python3-pip ffmpeg

# Install Python deps
RUN pip install faster-whisper websockets numpy torch

# Copy your server
WORKDIR /app
COPY server.py .

EXPOSE 8000
RUN export LD_LIBRARY_PATH=${PWD}/.venv/lib64/python3.11/site-packages/nvidia/cublas/lib:${PWD}/.venv/lib64/python3.11/site-packages/nvidia/cudnn/lib
CMD ["python3", "server.py"]
