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
    # Required by TEN-VAD
    libc++1 \
    # Python build tools
    build-essential \
    cmake \
    curl \
    ca-certificates \
    python3-dev \
    pkg-config \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv
RUN pip install --no-cache-dir uv

# Bundle ARM64 sources and runtime; compilation happens at container startup.
COPY docker/prepare-ten-vad.sh /usr/local/bin/prepare-ten-vad
RUN sh /usr/local/bin/prepare-ten-vad

# Keep the project and its virtual environment together.
WORKDIR /app

# Copy project files and install Python dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Create directories for temporary files and recordings
RUN mkdir -p /tmp/radiotelegram /app/recordings

# Copy application code
COPY radiotelegram/ ./radiotelegram/
COPY README.md ./

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app:/opt/ten-vad/build
ENV PYTHONUNBUFFERED=1
ENV AUDIO_DEVICE=hw:0,0
ENV AUDIO_OUTPUT_DEVICE=hw:0,0

# Default command
ENTRYPOINT ["python", "-m", "radiotelegram.docker_entrypoint"]
CMD ["python", "-m", "radiotelegram.main"]
