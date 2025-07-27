FROM python:3.12-slim

# Install runtime dependencies for audio processing
RUN apt-get update && apt-get install -y \
    # Audio libraries and tools
    ffmpeg \
    alsa-utils \
    libasound2-dev \
    # USB device utilities
    usbutils \
    # System utilities
    lsof \
    procps \
    # Python build tools
    build-essential \
    python3-dev \
    pkg-config \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files and install Python dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Create app directory
WORKDIR /app

# Create directories for temporary files and recordings
RUN mkdir -p /tmp/radiotelegram /app/recordings

# Copy application code
COPY radiotelegram/ ./radiotelegram/
COPY README.md ./

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV AUDIO_DEVICE=hw:1,0

# Default command
CMD ["uv", "run", "python", "-m", "radiotelegram.main"]
