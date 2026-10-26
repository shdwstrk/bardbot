FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install Python 3, pip, and essential audio libraries (like ffmpeg for audio decoding)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    build-essential \
    cmake \
    libopus-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install faster-whisper
RUN pip3 install --no-cache-dir --upgrade pip

# Copy only the requirements file first to take advantage of Docker caching layers
COPY requirements.txt .

# Install dependencies from the requirements file
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your transcription scripts into the container
COPY . .

CMD ["python3", "bot.py"]
