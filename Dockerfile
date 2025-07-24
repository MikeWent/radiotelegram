FROM python:3.12-slim

# Install runtime dependencies for audio processing
RUN apt-get update && apt-get install -y \
    # Audio libraries and tools
    ffmpeg \
    alsa-utils \
    pulseaudio-utils \
    libasound2-dev \
    libpulse-dev \
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

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create app directory and non-root user
WORKDIR /app
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

# Add user to audio group for audio device access
RUN usermod -a -G audio appuser

# Copy application code
COPY --chown=appuser:appuser radiotelegram/ ./radiotelegram/
COPY --chown=appuser:appuser README.md ./

# Create directories for temporary files and recordings
RUN mkdir -p /tmp/radiotelegram /app/recordings && \
    chown -R appuser:appuser /tmp/radiotelegram /app/recordings

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV AUDIO_DEVICE=pulse

# Default command
CMD ["python", "-m", "radiotelegram.main"]
