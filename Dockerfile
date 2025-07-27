FROM python:3.12-slim

# Install runtime dependencies for audio processing
RUN apt-get update && apt-get install -y \
    # Audio libraries and tools
    ffmpeg \
    alsa-utils \
    pulseaudio-utils \
    pipewire-pulse \
    pipewire-alsa \
    libasound2-dev \
    libpulse-dev \
    libpipewire-0.3-dev \
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

# Create app directory and non-root user
WORKDIR /app
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser -m

# Add user to audio group for audio device access
RUN usermod -a -G audio appuser

# Create directories for temporary files and recordings
RUN mkdir -p /tmp/radiotelegram /app/recordings /run/user/1000 /home/appuser/.cache/uv && \
    chown -R appuser:appuser /tmp/radiotelegram /app/recordings /run/user/1000 /home/appuser/.cache

# Copy application code
COPY --chown=appuser:appuser radiotelegram/ ./radiotelegram/
COPY --chown=appuser:appuser README.md ./

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV AUDIO_DEVICE=pulse

# Default command
CMD ["uv", "run", "python", "-m", "radiotelegram.main"]
