import asyncio
import datetime
import os
import queue
import select
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np
from scipy import signal
from scipy.io import wavfile

from radiotelegram.bus import MessageBus, Worker, cpu_intensive
from radiotelegram.events import (
    RxNoiseFloorStatsEvent,
    RxRecordingCompleteEvent,
    RxRecordingEndedEvent,
    RxRecordingStartedEvent,
    TxMessagePlaybackEndedEvent,
    TxMessagePlaybackStartedEvent,
)


class SpectralAnalyzer:
    """Provides spectral analysis for better squelch decisions."""

    def __init__(self, sample_rate: int = 48000, fft_size: int = 512):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.window = np.hanning(fft_size)
        self.freq_bins = np.fft.fftfreq(fft_size, 1 / sample_rate)

        # Voice frequency range (300Hz - 3.4kHz is typical for radio)
        self.voice_freq_low = 300
        self.voice_freq_high = 3400
        self.voice_bin_low = int(self.voice_freq_low * fft_size / sample_rate)
        self.voice_bin_high = int(self.voice_freq_high * fft_size / sample_rate)

        # More specific voice bands for better discrimination
        self.core_voice_low = 500  # Core voice frequencies
        self.core_voice_high = 2800
        self.core_voice_bin_low = int(self.core_voice_low * fft_size / sample_rate)
        self.core_voice_bin_high = int(self.core_voice_high * fft_size / sample_rate)

    @cpu_intensive
    def analyze_spectrum(
        self, audio_chunk: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        Analyze audio spectrum for better signal detection.

        Returns:
            voice_energy: Energy in voice frequency range
            total_energy: Total energy across spectrum
            spectral_centroid: Weighted average of frequencies
            voice_quality: Quality measure for voice-like characteristics
        """
        # Handle different chunk sizes by using the configured FFT size
        if len(audio_chunk) > self.fft_size:
            # Use first part of chunk for FFT analysis
            chunk_for_fft = audio_chunk[: self.fft_size]
        elif len(audio_chunk) < self.fft_size:
            # Pad chunk to FFT size
            chunk_for_fft = np.pad(
                audio_chunk, (0, self.fft_size - len(audio_chunk)), "constant"
            )
        else:
            chunk_for_fft = audio_chunk

        # Apply window and compute FFT
        windowed = chunk_for_fft * self.window
        fft = np.fft.fft(windowed)
        magnitude = np.abs(fft[: self.fft_size // 2])

        # Calculate voice band energy
        voice_energy = np.sum(magnitude[self.voice_bin_low : self.voice_bin_high] ** 2)
        total_energy = np.sum(magnitude**2)

        # Calculate core voice energy for better discrimination
        core_voice_energy = np.sum(
            magnitude[self.core_voice_bin_low : self.core_voice_bin_high] ** 2
        )

        # Spectral centroid (brightness measure)
        frequencies = self.freq_bins[: self.fft_size // 2]
        if total_energy > 0:
            spectral_centroid = np.sum(frequencies * magnitude**2) / total_energy
        else:
            spectral_centroid = 0

        # Voice quality measure - looks for typical voice spectral shape
        voice_quality = 0.0
        if total_energy > 1e-10:  # Avoid division by zero
            # Voice typically has more energy in mid frequencies than extremes
            low_band = np.sum(magnitude[1 : self.voice_bin_low] ** 2)  # Below voice
            mid_band = core_voice_energy  # Core voice frequencies
            high_band = np.sum(magnitude[self.voice_bin_high :] ** 2)  # Above voice

            # Good voice should have: mid > low and mid > high
            if mid_band > 0:
                low_ratio = low_band / (low_band + mid_band + high_band + 1e-10)
                high_ratio = high_band / (low_band + mid_band + high_band + 1e-10)
                mid_ratio = mid_band / (low_band + mid_band + high_band + 1e-10)

                # Voice quality is higher when mid frequencies dominate
                # and there's reasonable spectral spread (not pure tones)
                voice_quality = mid_ratio * (1.0 - abs(low_ratio - high_ratio))

                # Penalize pure tones (very narrow spectrum) - but be more permissive
                spectral_spread = (
                    np.sum(
                        (frequencies[: len(magnitude)] - spectral_centroid) ** 2
                        * magnitude**2
                    )
                    / total_energy
                )
                spread_factor = min(
                    1.0, spectral_spread / 500000
                )  # More permissive normalization (was 1000000)
                voice_quality *= spread_factor

        return voice_energy, total_energy, spectral_centroid, voice_quality


class AdaptiveNoiseFloor:
    """Adaptive noise floor estimation with multiple time constants."""

    def __init__(self, fast_alpha: float = 0.1, slow_alpha: float = 0.01):
        self.fast_alpha = fast_alpha  # Fast adaptation for quick changes
        self.slow_alpha = slow_alpha  # Slow adaptation for long-term average
        self.fast_noise_floor = -60.0
        self.slow_noise_floor = -60.0
        self.min_noise_floor = -80.0
        self.max_noise_floor = -30.0

    def update(self, current_db: float, is_signal_present: bool = False) -> float:
        """
        Update noise floor estimates.

        Args:
            current_db: Current audio level in dB
            is_signal_present: Whether a signal is currently detected

        Returns:
            Estimated noise floor in dB
        """
        if not is_signal_present:
            # Only update during silence periods
            self.fast_noise_floor = (
                self.fast_alpha * current_db
                + (1 - self.fast_alpha) * self.fast_noise_floor
            )
            self.slow_noise_floor = (
                self.slow_alpha * current_db
                + (1 - self.slow_alpha) * self.slow_noise_floor
            )

        # Clamp to reasonable bounds
        self.fast_noise_floor = np.clip(
            self.fast_noise_floor, self.min_noise_floor, self.max_noise_floor
        )
        self.slow_noise_floor = np.clip(
            self.slow_noise_floor, self.min_noise_floor, self.max_noise_floor
        )

        # Use average of fast and slow for final estimate
        return (self.fast_noise_floor + self.slow_noise_floor) / 2


class AdvancedSquelch:
    """Advanced squelch with multiple criteria for signal detection."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.noise_floor = AdaptiveNoiseFloor()
        self.spectral_analyzer = SpectralAnalyzer(sample_rate)

        # Squelch parameters - optimized based on voice/noise analysis
        self.level_threshold_db = 10.0  # dB above noise floor (balanced sensitivity)
        self.voice_energy_ratio = (
            0.15  # Min ratio of voice energy to total (relaxed for voice)
        )
        self.spectral_centroid_min = 200  # Hz (wide range for various voices)
        self.spectral_centroid_max = 2500  # Hz (wide range for various voices)
        self.min_voice_quality = (
            0.02  # Minimum voice quality score (moderate requirement)
        )
        self.min_absolute_level = -40.0  # Absolute minimum level in dB (sensitive)

        # Hysteresis for stability - optimized for immediate response
        self.open_threshold = 0.65  # Reduced from 0.75 for faster opening
        self.close_threshold = (
            0.55  # Reduced gap for faster response while maintaining stability
        )
        self.is_open = False

    @cpu_intensive
    def process(self, audio_chunk: np.ndarray) -> Tuple[bool, dict]:
        """
        Process audio chunk and determine if squelch should be open.

        Returns:
            is_open: Whether squelch should be open
            stats: Dictionary with analysis statistics
        """
        # Basic level analysis
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
        current_db = 20 * np.log10(rms / 32767.0 + 1e-6)

        # Update noise floor (don't update if currently open)
        noise_floor_db = self.noise_floor.update(current_db, self.is_open)
        level_margin_db = current_db - noise_floor_db

        # Spectral analysis
        voice_energy, total_energy, spectral_centroid, voice_quality = (
            self.spectral_analyzer.analyze_spectrum(audio_chunk)
        )

        # Calculate criteria scores (0-1)
        level_score = max(0, min(1, (level_margin_db - 3) / 15))  # Smooth 3-18dB range

        if total_energy > 0:
            voice_ratio_score = min(
                1, voice_energy / total_energy / self.voice_energy_ratio
            )
        else:
            voice_ratio_score = 0

        # Spectral centroid score (peak around speech frequencies)
        if (
            self.spectral_centroid_min
            <= spectral_centroid
            <= self.spectral_centroid_max
        ):
            centroid_score = 1.0
        else:
            centroid_score = 0.5  # Partial credit outside ideal range

        # Voice quality score (0-1, higher is more voice-like)
        quality_score = min(1.0, voice_quality / self.min_voice_quality)

        # Absolute level check - reject very weak signals regardless of other criteria
        if current_db < self.min_absolute_level:
            level_score = 0

        # Balanced weighted scoring optimized for voice/noise discrimination
        combined_score = (
            0.35 * level_score  # Signal strength above noise floor
            + 0.25 * voice_ratio_score  # Energy concentration in voice bands
            + 0.20 * centroid_score  # Spectral brightness in voice range
            + 0.20 * quality_score  # Voice-like spectral characteristics
        )

        # Apply hysteresis
        threshold = self.close_threshold if self.is_open else self.open_threshold
        self.is_open = combined_score >= threshold

        stats = {
            "current_db": current_db,
            "noise_floor_db": noise_floor_db,
            "level_margin_db": level_margin_db,
            "voice_energy": voice_energy,
            "total_energy": total_energy,
            "spectral_centroid": spectral_centroid,
            "voice_quality": voice_quality,
            "level_score": level_score,
            "voice_ratio_score": voice_ratio_score,
            "centroid_score": centroid_score,
            "quality_score": quality_score,
            "combined_score": combined_score,
            "squelch_open": self.is_open,
        }

        return self.is_open, stats


class VoiceDetector:
    """Analyzes recorded audio to determine if it contains actual voice content."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.spectral_analyzer = SpectralAnalyzer(sample_rate)

        # Voice detection thresholds (optimized for weak signal detection)
        self.min_voice_energy_ratio = 0.1  # Even more permissive for weak voice signals
        self.min_voice_quality_score = 0.1  # Reduced for weak signals
        self.min_spectral_centroid = 250  # Hz - slightly lower for weak voices
        self.max_spectral_centroid = 2800  # Hz - slightly higher range
        self.min_analysis_duration = 0.5  # Minimum seconds to analyze
        self.voice_consistency_threshold = (
            0.02  # Even lower threshold for very weak signals
        )

        # Key discriminator: spectral variability (voice has higher variability than noise)
        # Relaxed significantly to catch consistent voice signals
        self.min_spectral_variability = (
            0.1  # Much lower - allow very consistent spectral content (was 0.5)
        )
        self.max_spectral_variability = 3.0  # Slightly higher tolerance (was 2.0)

        # Energy and dynamics thresholds - relaxed for weak signals
        self.min_energy_db = -100.0  # Effectively disabled for weak signal detection
        self.min_dynamic_range_db = 2.0  # Further reduced for weak signals (was 3.0)

    @cpu_intensive
    def analyze_recording(self, filepath: str) -> Tuple[bool, dict]:
        """
        Analyze a recording to determine if it contains voice.

        Returns:
            is_voice: True if recording contains voice
            analysis: Dictionary with detailed analysis results
        """
        try:
            # Load audio file
            if filepath.endswith(".wav"):
                sample_rate, audio_data = wavfile.read(filepath)
            else:
                # For other formats, use ffmpeg to convert to wav temporarily
                temp_wav = filepath.replace(".ogg", "_temp.wav")
                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        filepath,
                        "-ar",
                        str(self.sample_rate),
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_s16le",
                        temp_wav,
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    return False, {"error": "Failed to convert audio"}

                sample_rate, audio_data = wavfile.read(temp_wav)
                os.remove(temp_wav)

            # Ensure mono
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]

            # Convert to float32 and normalize
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32767.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483647.0

            duration = len(audio_data) / sample_rate

            if duration < self.min_analysis_duration:
                return False, {"error": "Recording too short for analysis"}

            # Analyze in chunks to get consistent measurements
            chunk_size = int(0.1 * sample_rate)  # 100ms chunks
            voice_chunks = 0
            total_chunks = 0

            energy_levels = []
            voice_ratios = []
            centroids = []
            voice_qualities = []
            sustained_energy_chunks = (
                0  # Count chunks with sustained energy (not clicks)
            )

            for i in range(0, len(audio_data) - chunk_size, chunk_size):
                chunk = audio_data[i : i + chunk_size]

                # Convert back to int16 for spectral analyzer
                chunk_int16 = (chunk * 32767).astype(np.int16)

                # Get spectral analysis
                voice_energy, total_energy, spectral_centroid, voice_quality = (
                    self.spectral_analyzer.analyze_spectrum(chunk_int16)
                )

                # Calculate energy in dB
                rms = np.sqrt(np.mean(chunk**2))
                energy_db = 20 * np.log10(rms + 1e-10)

                energy_levels.append(energy_db)

                # Calculate voice energy ratio
                if total_energy > 0:
                    voice_ratio = voice_energy / total_energy
                else:
                    voice_ratio = 0

                voice_ratios.append(voice_ratio)
                centroids.append(spectral_centroid)
                voice_qualities.append(voice_quality)

                # Analyze energy distribution within chunk to detect clicks vs sustained voice
                # Clicks have sharp energy spikes, voice has more sustained energy
                chunk_abs = np.abs(chunk)
                energy_variance = np.var(chunk_abs)
                energy_mean = np.mean(chunk_abs)
                energy_cv = energy_variance / (
                    energy_mean**2 + 1e-10
                )  # Coefficient of variation

                # Sustained energy check (lower coefficient of variation = more sustained)
                # Clicks have high variance relative to mean, voice is more consistent
                max_energy_cv_for_voice = 3.0  # More permissive threshold for sustained vs impulsive energy (was 2.0)
                is_sustained = energy_cv < max_energy_cv_for_voice

                if is_sustained and energy_db > self.min_energy_db:
                    sustained_energy_chunks += 1

                # Check if this chunk passes voice criteria
                chunk_is_voice = (
                    energy_db > self.min_energy_db
                    and voice_ratio > self.min_voice_energy_ratio
                    and voice_quality > self.min_voice_quality_score
                    and self.min_spectral_centroid
                    <= spectral_centroid
                    <= self.max_spectral_centroid
                    and is_sustained  # Add sustained energy requirement
                )

                if chunk_is_voice:
                    voice_chunks += 1
                total_chunks += 1

            # Calculate overall statistics
            if not energy_levels:
                return False, {"error": "No audio chunks to analyze"}

            avg_energy_db = np.mean(energy_levels)
            max_energy_db = np.max(energy_levels)
            min_energy_db = np.min(energy_levels)
            dynamic_range_db = max_energy_db - min_energy_db

            avg_voice_ratio = np.mean(voice_ratios)
            avg_centroid = np.mean(centroids)
            avg_voice_quality = np.mean(voice_qualities)

            # Calculate spectral stability (lower = more stable/noise-like)
            centroid_std = np.std(centroids) if len(centroids) > 1 else 0
            spectral_stability = centroid_std / (
                avg_centroid + 1
            )  # Normalized standard deviation

            voice_consistency = voice_chunks / total_chunks if total_chunks > 0 else 0

            # Calculate sustained energy ratio (important for distinguishing clicks from voice)
            sustained_energy_ratio = (
                sustained_energy_chunks / total_chunks if total_chunks > 0 else 0
            )
            min_sustained_energy_ratio = (
                0.02  # Even lower threshold for very weak signals (was 0.05)
            )

            # Enhanced voice detection criteria
            # Voice should have significant spectral variability (not like pure noise or tones)
            spectral_variability_ok = (
                self.min_spectral_variability
                <= spectral_stability
                <= self.max_spectral_variability
            )

            # Final voice detection decision with enhanced criteria
            is_voice = (
                avg_energy_db > self.min_energy_db
                and dynamic_range_db > self.min_dynamic_range_db
                and avg_voice_ratio > self.min_voice_energy_ratio
                and avg_voice_quality > self.min_voice_quality_score
                and self.min_spectral_centroid
                <= avg_centroid
                <= self.max_spectral_centroid
                and voice_consistency > self.voice_consistency_threshold
                and spectral_variability_ok  # Key discriminator: voice has moderate spectral changes
                and sustained_energy_ratio
                > min_sustained_energy_ratio  # Reject clicks/pops
            )

            analysis = {
                "duration": duration,
                "avg_energy_db": avg_energy_db,
                "dynamic_range_db": dynamic_range_db,
                "avg_voice_ratio": avg_voice_ratio,
                "avg_spectral_centroid": avg_centroid,
                "avg_voice_quality": avg_voice_quality,
                "voice_consistency": voice_consistency,
                "spectral_stability": spectral_stability,
                "sustained_energy_ratio": sustained_energy_ratio,
                "sustained_energy_chunks": sustained_energy_chunks,
                "voice_chunks": voice_chunks,
                "total_chunks": total_chunks,
                "is_voice": is_voice,
            }

            return bool(is_voice), analysis

        except Exception as e:
            return False, {"error": f"Analysis failed: {str(e)}"}


class EnhancedRxListenWorker(Worker):
    """Enhanced receiver worker with advanced audio processing."""

    def __init__(
        self, bus: MessageBus, sample_rate=48000, chunk_size=256, audio_device="pulse"
    ):
        super().__init__(bus)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size  # Reduced default chunk size for lower latency
        self.audio_device = (
            audio_device  # Default to pulse (PipeWire/PulseAudio compatible)
        )

        # Recording parameters optimized for responsive squelch
        self.silence_duration = 2.0  # Seconds to wait before stopping recording
        self.minimal_recording_duration = 1.0  # Min duration after trimming
        self.enabled = True
        self.recording = False
        self.process = None
        self.silence_counter = 0

        # Squelch state tracking - optimized for immediate response
        self.squelch_open_time: Optional[float] = None
        self.min_squelch_open_duration = 0.5

        # Pre-recording buffer to capture audio before squelch opens
        self.pre_record_duration = 0.5  # Seconds of audio to buffer before squelch
        self.pre_record_buffer = []  # Circular buffer for audio chunks
        self.max_pre_record_chunks = int(
            self.pre_record_duration * sample_rate / chunk_size
        )

        # File paths
        self.recording_filepath: Optional[str] = None
        self.pre_record_filepath: Optional[str] = None
        self.recording_start_time: Optional[datetime.datetime] = None

        # Advanced audio processing
        self.squelch = AdvancedSquelch(sample_rate)
        self.voice_detector = VoiceDetector(sample_rate)

        # Statistics and monitoring - reduced interval for better responsiveness monitoring
        self.stats_interval = 2.0  # Publish stats every 2 seconds (reduced from 5s)
        self.last_stats_publish_time = 0

        # Performance monitoring for latency debugging
        self.chunk_process_times = []
        self.max_process_time_samples = 100  # Keep last 100 samples

        # Store the event loop for thread-safe async operations
        try:
            self.event_loop = asyncio.get_event_loop()
        except RuntimeError:
            self.event_loop = None

        # Event subscriptions
        self.bus.subscribe(TxMessagePlaybackStartedEvent, self.on_playback_started)
        self.bus.subscribe(TxMessagePlaybackEndedEvent, self.on_playback_finished)

        # Test audio device availability at startup
        self._test_audio_device()

    async def on_playback_started(self, event: TxMessagePlaybackStartedEvent):
        """Disable listening during playback to prevent feedback."""
        self.enabled = False
        if self.recording:
            self.logger.info("Stopping recording due to TX playback starting")
            self.stop_recording()

    async def on_playback_finished(self, event: TxMessagePlaybackEndedEvent):
        """Re-enable listening after playback ends."""
        self.enabled = True
        # Restart audio stream processing
        self.logger.info("TX playback finished, restarting audio stream processing")
        asyncio.create_task(asyncio.to_thread(self.process_audio_stream))

    def _test_audio_device(self):
        """Test audio device availability and log diagnostics."""
        try:
            # First check if PipeWire is running
            self._check_pipewire_status()

            # Test if we can list audio devices
            result = subprocess.run(
                ["ffmpeg", "-f", "alsa", "-list_devices", "true", "-i", "dummy"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.stderr:
                self.logger.debug(f"Available ALSA devices:\n{result.stderr}")

            # Extract available input devices from the output
            available_devices = self._parse_alsa_devices(result.stderr)
            self.logger.info(f"Detected ALSA input devices: {available_devices}")

            # Test the configured device first
            if self._test_device(self.audio_device):
                self.logger.info(f"Audio device '{self.audio_device}' is working")
                return

            # If default device fails, try other common device names
            fallback_devices = [
                "pulse",  # PulseAudio/PipeWire compatibility layer (try first)
                "pipewire",  # Direct PipeWire access
                "hw:0,0",
                "plughw:0,0",
                "hw:1,0",
                "plughw:1,0",
                "hw:2,0",
                "plughw:2,0",  # Try more hardware devices
                "hw:0",
                "plughw:0",  # Simplified hardware references
                "hw:1",
                "plughw:1",
            ]

            for device in fallback_devices:
                if device != self.audio_device:  # Don't test the same device twice
                    self.logger.info(f"Testing fallback audio device: {device}")
                    if self._test_device(device):
                        self.logger.info(f"Switching to working audio device: {device}")
                        self.audio_device = device
                        return

            # If no device works, log the problem and provide diagnostics
            self._log_troubleshooting_info()
            self._run_audio_diagnostics()

        except subprocess.TimeoutExpired:
            self.logger.error("Audio device test timed out")
        except Exception as e:
            self.logger.error(f"Audio device test error: {e}")

    def _run_audio_diagnostics(self):
        """Run comprehensive audio diagnostics to help troubleshoot issues."""
        self.logger.info("Running audio diagnostics...")

        try:
            # Check what processes are using audio
            lsof_result = subprocess.run(
                ["sudo", "lsof", "/dev/snd/*"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if lsof_result.returncode == 0 and lsof_result.stdout:
                self.logger.info(
                    f"Processes using audio devices:\n{lsof_result.stdout}"
                )
            else:
                self.logger.info(
                    "No processes found using audio devices (or lsof failed)"
                )

        except subprocess.TimeoutExpired:
            self.logger.warning("Audio diagnostics lsof command timed out")
        except Exception as e:
            self.logger.debug(f"Could not run lsof diagnostics: {e}")

        try:
            # List available ALSA devices
            arecord_result = subprocess.run(
                ["arecord", "-l"], capture_output=True, text=True, timeout=10
            )
            if arecord_result.returncode == 0:
                self.logger.info(f"ALSA recording devices:\n{arecord_result.stdout}")
            else:
                self.logger.warning(f"arecord -l failed: {arecord_result.stderr}")

        except subprocess.TimeoutExpired:
            self.logger.warning("arecord command timed out")
        except Exception as e:
            self.logger.debug(f"Could not run arecord diagnostics: {e}")

        try:
            # Check user groups
            groups_result = subprocess.run(
                ["groups"], capture_output=True, text=True, timeout=5
            )
            if groups_result.returncode == 0:
                groups = groups_result.stdout.strip()
                self.logger.info(f"User groups: {groups}")
                if "audio" not in groups:
                    self.logger.warning(
                        "User is not in 'audio' group - this may cause permission issues"
                    )
            else:
                self.logger.debug("Could not check user groups")

        except Exception as e:
            self.logger.debug(f"Could not check user groups: {e}")

        try:
            # Check PipeWire services status
            pipewire_status = subprocess.run(
                ["systemctl", "--user", "is-active", "pipewire"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            pulse_status = subprocess.run(
                ["systemctl", "--user", "is-active", "pipewire-pulse"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            self.logger.info(
                f"PipeWire service status: {pipewire_status.stdout.strip()}"
            )
            self.logger.info(
                f"PipeWire-Pulse service status: {pulse_status.stdout.strip()}"
            )

        except Exception as e:
            self.logger.debug(f"Could not check PipeWire service status: {e}")

    def _check_pipewire_status(self):
        """Check if PipeWire is running and log status."""
        try:
            # Check if PipeWire daemon is running
            pipewire_result = subprocess.run(
                ["pgrep", "-f", "pipewire"], capture_output=True, text=True
            )

            if pipewire_result.returncode == 0:
                self.logger.info("PipeWire daemon is running")

                # Try to get PipeWire-pulse status
                try:
                    pulse_result = subprocess.run(
                        ["pgrep", "-f", "pipewire-pulse"],
                        capture_output=True,
                        text=True,
                    )
                    if pulse_result.returncode == 0:
                        self.logger.info(
                            "PipeWire-PulseAudio compatibility layer is running"
                        )
                except:
                    pass

            else:
                self.logger.warning(
                    "PipeWire daemon not detected, checking for PulseAudio..."
                )

                # Check for PulseAudio
                pulse_result = subprocess.run(
                    ["pgrep", "-f", "pulseaudio"], capture_output=True, text=True
                )

                if pulse_result.returncode == 0:
                    self.logger.info("PulseAudio daemon is running")
                else:
                    self.logger.warning("Neither PipeWire nor PulseAudio detected")

        except Exception as e:
            self.logger.debug(f"Could not check PipeWire/PulseAudio status: {e}")

    def _log_troubleshooting_info(self):
        """Log comprehensive troubleshooting information."""
        self.logger.error("No working audio input device found!")
        self.logger.error("PipeWire/PulseAudio troubleshooting:")
        self.logger.error(
            "1. Check if PipeWire is running: 'systemctl --user status pipewire'"
        )
        self.logger.error(
            "2. Start PipeWire: 'systemctl --user start pipewire pipewire-pulse'"
        )
        self.logger.error("3. Check what's using audio: 'sudo lsof /dev/snd/*'")
        self.logger.error(
            "4. Stop other audio applications (browsers, media players, etc.)"
        )
        self.logger.error(
            "5. Restart audio services: 'systemctl --user restart pipewire'"
        )
        self.logger.error("6. Check if you're in audio group: 'groups $USER'")
        self.logger.error("7. Verify hardware: 'arecord -l' and 'aplay -l'")
        self.logger.error("8. For ALSA fallback: 'sudo alsa force-reload'")
        self.logger.error("9. Check for exclusive mode: kill other audio processes")

    @cpu_intensive
    def _add_to_pre_record_buffer(self, audio_chunk: np.ndarray):
        """Add audio chunk to the pre-recording circular buffer."""
        # Convert to bytes for consistent storage
        audio_bytes = audio_chunk.tobytes()

        # Add to circular buffer
        self.pre_record_buffer.append(audio_bytes)

        # Maintain buffer size
        if len(self.pre_record_buffer) > self.max_pre_record_chunks:
            self.pre_record_buffer.pop(0)

    @cpu_intensive
    def _save_pre_record_buffer(self, filepath: str):
        """Save the pre-recording buffer to a WAV file."""
        if not self.pre_record_buffer:
            self.logger.warning("No pre-record buffer to save")
            return False

        try:
            # Combine all buffered chunks
            combined_bytes = b"".join(self.pre_record_buffer)

            # Convert bytes back to numpy array
            audio_data = np.frombuffer(combined_bytes, dtype=np.int16)

            # Save as WAV file
            wavfile.write(filepath, self.sample_rate, audio_data)

            file_size = os.path.getsize(filepath)
            duration = len(audio_data) / self.sample_rate
            self.logger.debug(
                f"Saved pre-record buffer: {file_size} bytes, {duration:.3f}s"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to save pre-record buffer: {e}")
            return False

    def _parse_alsa_devices(self, stderr_output):
        """Parse FFmpeg ALSA device list output to extract available devices."""
        devices = []
        if not stderr_output:
            return devices

        lines = stderr_output.split("\n")
        for line in lines:
            # Look for input device lines like "[0] card 0, device 0: ..."
            if "Input" in line and "card" in line:
                # Extract device identifier
                if "[" in line and "]" in line:
                    device_info = line.split("]", 1)[1].strip()
                    devices.append(device_info)
        return devices

    def _test_device(self, device):
        """Test if a specific audio device works by recording a short sample."""
        try:
            test_file = "/tmp/audio_test.wav"
            test_command = [
                "ffmpeg",
                "-y",
                "-f",
                "alsa",
                "-i",
                device,
                "-t",
                "0.1",  # Record for 0.1 seconds
                "-ar",
                str(self.sample_rate),
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                test_file,
            ]

            test_result = subprocess.run(
                test_command, capture_output=True, text=True, timeout=5
            )

            if test_result.returncode == 0 and os.path.exists(test_file):
                file_size = os.path.getsize(test_file)
                self.logger.debug(
                    f"Device '{device}' test successful - created {file_size} byte test file"
                )
                os.remove(test_file)
                return True
            else:
                # Parse the error to provide better diagnostics
                error_msg = test_result.stderr.lower()
                if "device or resource busy" in error_msg:
                    self.logger.warning(
                        f"Device '{device}' is busy (being used by another process)"
                    )
                elif "no such file or directory" in error_msg:
                    self.logger.debug(f"Device '{device}' does not exist")
                elif "input/output error" in error_msg:
                    self.logger.debug(
                        f"Device '{device}' has I/O error (may be busy or misconfigured)"
                    )
                elif "permission denied" in error_msg:
                    self.logger.warning(
                        f"Device '{device}' permission denied (check audio group membership)"
                    )
                else:
                    self.logger.debug(
                        f"Device '{device}' test failed: {test_result.stderr.strip()}"
                    )
                return False

        except subprocess.TimeoutExpired:
            self.logger.debug(f"Device '{device}' test timed out")
            return False
        except Exception as e:
            self.logger.debug(f"Device '{device}' test error: {e}")
            return False

    def start_recording(self):
        """Start recording with enhanced processing and pre-record buffer."""
        if self.recording:
            return

        os.makedirs("/tmp", exist_ok=True)
        timestamp = datetime.datetime.now().timestamp()

        # First, save the pre-record buffer
        pre_record_filename = f"pre_record_{timestamp:.3f}.wav"
        pre_record_filepath = os.path.join("/tmp", pre_record_filename)

        # Save buffered audio to catch the beginning of transmission
        if self._save_pre_record_buffer(pre_record_filepath):
            self.logger.info(f"Pre-record buffer saved: {pre_record_filepath}")

        # Create main recording file
        filename = f"recording_{timestamp:.3f}.wav"
        self.recording_filepath = os.path.join("/tmp", filename)
        self.recording_start_time = datetime.datetime.now()

        # Record raw audio first, then process
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "alsa",
            "-i",
            self.audio_device,  # Use the tested/detected audio device
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-c:a",
            "pcm_s16le",  # Raw PCM for processing
            self.recording_filepath,
        ]

        self.logger.debug(
            f"Starting FFmpeg recording with command: {' '.join(command)}"
        )

        try:
            self.process = subprocess.Popen(
                command, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True
            )

            # Give FFmpeg a moment to initialize
            time.sleep(0.1)

            # Check if process is still running (didn't fail immediately)
            if self.process.poll() is not None:
                # Process has already terminated
                _, stderr = self.process.communicate()
                self.logger.error(f"FFmpeg failed to start recording: {stderr}")
                self.recording = False
                # Clean up pre-record file if recording failed
                if os.path.exists(pre_record_filepath):
                    os.remove(pre_record_filepath)
                return

            self.recording = True
            self.silence_counter = 0

            # Store pre-record filepath for later merging
            self.pre_record_filepath = (
                pre_record_filepath if os.path.exists(pre_record_filepath) else None
            )

            asyncio.run(self.bus.publish(RxRecordingStartedEvent()))
            self.logger.info(f"Enhanced recording started: {self.recording_filepath}")
            if self.pre_record_filepath:
                self.logger.info(
                    f"Pre-record buffer will be merged: {self.pre_record_filepath}"
                )
        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            self.recording = False
            # Clean up pre-record file if recording failed
            if os.path.exists(pre_record_filepath):
                os.remove(pre_record_filepath)

    def stop_recording(self):
        """Stop recording and apply advanced processing."""
        if not self.recording or self.recording_start_time is None:
            return

        recording_duration = (
            datetime.datetime.now() - self.recording_start_time
        ).total_seconds()

        # Stop ffmpeg process
        if self.process:
            try:
                self.process.terminate()
                _, stderr = self.process.communicate(timeout=5.0)
                if stderr and self.process.returncode != 0:
                    self.logger.error(f"FFmpeg recording error: {stderr}")
            except subprocess.TimeoutExpired:
                self.logger.warning("FFmpeg process termination timeout, force killing")
                self.process.kill()
                try:
                    _, stderr = self.process.communicate(timeout=2.0)
                    if stderr:
                        self.logger.error(f"FFmpeg recording error (killed): {stderr}")
                except subprocess.TimeoutExpired:
                    self.logger.error("FFmpeg process could not be killed")
            finally:
                self.process = None

        self.recording = False
        asyncio.run(self.bus.publish(RxRecordingEndedEvent()))

        # Give FFmpeg a moment to finalize the file
        time.sleep(0.1)

        # Check if recording file actually exists
        if not self.recording_filepath or not os.path.exists(self.recording_filepath):
            self.logger.error(
                f"Recording file does not exist: {self.recording_filepath}"
            )
            return

        # Check file size to ensure we have actual audio data
        try:
            file_size = os.path.getsize(self.recording_filepath)
            self.logger.debug(f"Recording file size: {file_size} bytes")
            if file_size < 1024:  # Less than 1KB is probably an empty or corrupt file
                self.logger.warning(
                    f"Recording file is too small ({file_size} bytes), likely corrupt"
                )
                os.remove(self.recording_filepath)
                return
        except OSError as e:
            self.logger.error(f"Cannot access recording file: {e}")
            return

        # Calculate trimmed duration
        trimmed_duration = max(0, recording_duration - self.silence_duration)

        if trimmed_duration > self.minimal_recording_duration:
            self._process_recording(trimmed_duration)
        else:
            self.logger.info(
                f"Recording too short ({trimmed_duration:.3f}s), discarded"
            )
            if self.recording_filepath and os.path.exists(self.recording_filepath):
                os.remove(self.recording_filepath)

    @cpu_intensive
    def _process_recording(self, duration: float):
        """Apply advanced processing to recorded audio with pre-record buffer merge."""
        if not self.recording_filepath:
            return

        # Double-check that the recording file exists
        if not os.path.exists(self.recording_filepath):
            self.logger.error(
                f"Recording file missing for processing: {self.recording_filepath}"
            )
            return

        try:
            # Step 1: Merge pre-record buffer with main recording if available
            merged_filepath = self.recording_filepath

            if self.pre_record_filepath and os.path.exists(self.pre_record_filepath):
                # Create a merged file that includes the pre-record buffer
                merged_filepath = self.recording_filepath.replace(".wav", "_merged.wav")

                # Use FFmpeg to concatenate pre-record buffer and main recording
                concat_command = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    self.pre_record_filepath,
                    "-i",
                    self.recording_filepath,
                    "-filter_complex",
                    "[0:a][1:a]concat=n=2:v=0:a=1[a]",
                    "-map",
                    "[a]",
                    "-c:a",
                    "pcm_s16le",
                    merged_filepath,
                ]

                self.logger.debug(
                    f"Merging pre-record buffer with command: {' '.join(concat_command)}"
                )

                merge_result = subprocess.run(
                    concat_command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=15,
                )

                if merge_result.returncode == 0:
                    self.logger.info(
                        f"Successfully merged pre-record buffer with main recording"
                    )
                    # Update duration to include pre-record buffer
                    try:
                        sample_rate, merged_audio_data = wavfile.read(merged_filepath)
                        total_duration = len(merged_audio_data) / sample_rate
                        self.logger.info(
                            f"Total merged duration: {total_duration:.3f}s (was {duration:.3f}s)"
                        )
                        duration = total_duration
                    except Exception as e:
                        self.logger.warning(f"Could not read merged file duration: {e}")
                        duration += self.pre_record_duration  # Estimate
                else:
                    self.logger.error(
                        f"Failed to merge pre-record buffer: {merge_result.stderr}"
                    )
                    # Fall back to using original recording without pre-record buffer
                    merged_filepath = self.recording_filepath

                # Clean up pre-record buffer file
                try:
                    os.remove(self.pre_record_filepath)
                except:
                    pass

            processed_filepath = merged_filepath.replace(".wav", "_processed.ogg")

            # Advanced FFmpeg filter chain for radio processing
            filter_complex = (
                # Note: We don't trim the beginning anymore since we want to keep the pre-record buffer
                "[0:a]"
                # 1. High-pass filter to remove low-frequency noise
                "highpass=f=300,"
                # 2. De-emphasis (radio typically has pre-emphasis)
                "treble=g=-6:f=1000:width_type=h:width=1000,"
                # 3. Bandpass filter for voice clarity
                "lowpass=f=3400,"
                # 4. Dynamic range compression for consistent levels
                "compand=attacks=0.01:decays=0.5:points=-70/-70|-60/-50|-30/-20|-10/-10:soft-knee=6:gain=0:volume=-20,"
                # 5. Noise reduction (gentle)
                "afftdn=nr=10:nf=-50:tn=1,"
                # 6. Final limiter to prevent clipping
                "alimiter=level_in=1:level_out=1:limit=-1dB:attack=5:release=50[a]"
            )

            command = [
                "ffmpeg",
                "-y",
                "-i",
                merged_filepath,
                "-filter_complex",
                filter_complex,
                "-map",
                "[a]",
                "-ar",
                str(self.sample_rate),
                "-c:a",
                "libopus",
                "-b:a",
                "64k",  # Good quality for voice
                processed_filepath,
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                # Step 1: Analyze the processed recording for voice content
                is_voice, analysis = self.voice_detector.analyze_recording(
                    processed_filepath
                )

                self.logger.info(
                    f"Voice analysis - "
                    f"Contains voice: {is_voice}, "
                    f"Energy: {analysis.get('avg_energy_db', 0):.1f}dB, "
                    f"Voice ratio: {analysis.get('avg_voice_ratio', 0):.3f}, "
                    f"Voice quality: {analysis.get('avg_voice_quality', 0):.3f}, "
                    f"Consistency: {analysis.get('voice_consistency', 0):.1%}, "
                    f"Sustained: {analysis.get('sustained_energy_ratio', 0):.1%}, "
                    f"Spectral stability: {analysis.get('spectral_stability', 0):.3f}, "
                    f"Centroid: {analysis.get('avg_spectral_centroid', 0):.0f}Hz"
                )

                if is_voice:
                    # Voice detected - publish the processed recording
                    asyncio.run(
                        self.bus.publish(
                            RxRecordingCompleteEvent(filepath=processed_filepath)
                        )
                    )
                    self.logger.info(
                        f"Voice recording completed and sent: {processed_filepath}"
                    )
                else:
                    # No voice detected - discard the recording
                    self.logger.info(
                        f"No voice detected, discarding recording: {processed_filepath}"
                    )
                    if os.path.exists(processed_filepath):
                        os.remove(processed_filepath)
            else:
                self.logger.error(f"FFmpeg processing failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            self.logger.error("Recording processing timeout")
        except Exception as e:
            self.logger.error(f"Error processing recording: {e}")
        finally:
            # Clean up recording files
            if os.path.exists(self.recording_filepath):
                os.remove(self.recording_filepath)

            # Clean up merged file if it was created
            merged_filepath = self.recording_filepath.replace(".wav", "_merged.wav")
            if merged_filepath != self.recording_filepath and os.path.exists(
                merged_filepath
            ):
                os.remove(merged_filepath)

    def _process_audio_chunk_immediate(self, audio_chunk: np.ndarray):
        """
        Process audio chunk with immediate response (synchronous squelch processing).
        This method prioritizes speed over CPU efficiency for immediate squelch response.
        """
        start_time = time.time()
        try:
            # Always add to pre-record buffer (unless we're already recording)
            if not self.recording:
                self._add_to_pre_record_buffer(audio_chunk)

            # Call squelch processing directly (synchronously) for immediate response
            # This bypasses the @cpu_intensive threading to eliminate latency
            squelch_open, stats = self._immediate_squelch_process(audio_chunk)

            # Handle squelch state changes immediately
            current_time = time.time()

            if squelch_open and not self.recording:
                # Squelch opened - start recording immediately (no delay needed now)
                self.start_recording()
                self.squelch_open_time = None
                self.logger.info("Recording started immediately on squelch open")

            elif not squelch_open:
                # Squelch closed
                if self.squelch_open_time is not None:
                    self.logger.debug("Squelch closed before recording threshold")
                self.squelch_open_time = None

                if self.recording:
                    self.silence_counter += 1
                    chunks_for_silence = self.silence_duration * (
                        self.sample_rate / self.chunk_size
                    )

                    if self.silence_counter >= chunks_for_silence:
                        self.stop_recording()
            else:
                # Squelch is open and recording
                self.silence_counter = 0

            # Track processing performance
            process_time = time.time() - start_time
            self.chunk_process_times.append(process_time)
            if len(self.chunk_process_times) > self.max_process_time_samples:
                self.chunk_process_times.pop(0)

            # Schedule statistics publishing asynchronously (non-blocking)
            if (
                current_time - self.last_stats_publish_time >= self.stats_interval
                and self.event_loop
            ):
                # Add performance stats
                avg_process_time = sum(self.chunk_process_times) / len(
                    self.chunk_process_times
                )
                max_process_time = max(self.chunk_process_times)
                stats["avg_process_time_ms"] = avg_process_time * 1000
                stats["max_process_time_ms"] = max_process_time * 1000

                # Use the stored event loop reference for thread-safe async operations
                asyncio.run_coroutine_threadsafe(
                    self._publish_enhanced_stats(stats), self.event_loop
                )
                self.last_stats_publish_time = current_time

        except Exception as e:
            self.logger.error(f"Error in immediate audio chunk processing: {e}")

    def _immediate_squelch_process(self, audio_chunk: np.ndarray) -> Tuple[bool, dict]:
        """
        Immediate squelch processing without threading for minimal latency.
        This is a simplified version of the squelch processing optimized for speed.
        """
        try:
            # Basic level analysis (fast)
            rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
            current_db = 20 * np.log10(rms / 32767.0 + 1e-6)

            # Update noise floor (don't update if currently open)
            noise_floor_db = self.squelch.noise_floor.update(
                current_db, self.squelch.is_open
            )
            level_margin_db = current_db - noise_floor_db

            # Simplified spectral analysis for speed
            if (
                len(audio_chunk) >= 256
            ):  # Only do spectral analysis if we have enough samples
                # Use a smaller FFT for speed
                fft_chunk = audio_chunk[:256] if len(audio_chunk) > 256 else audio_chunk
                voice_energy, total_energy, spectral_centroid, voice_quality = (
                    self.squelch.spectral_analyzer.analyze_spectrum(fft_chunk)
                )
            else:
                # Skip spectral analysis for very small chunks
                voice_energy = 0
                total_energy = 1
                spectral_centroid = 1000  # Assume reasonable default
                voice_quality = 0

            # Simplified scoring for immediate response
            level_score = max(
                0, min(1, (level_margin_db - 3) / 15)
            )  # Smooth 3-18dB range

            if total_energy > 0:
                voice_ratio_score = min(
                    1, voice_energy / total_energy / self.squelch.voice_energy_ratio
                )
            else:
                voice_ratio_score = 0

            # Simplified spectral centroid score
            if (
                self.squelch.spectral_centroid_min
                <= spectral_centroid
                <= self.squelch.spectral_centroid_max
            ):
                centroid_score = 1.0
            else:
                centroid_score = 0.5

            # Simplified voice quality score
            quality_score = min(1.0, voice_quality / self.squelch.min_voice_quality)

            # Absolute level check
            if current_db < self.squelch.min_absolute_level:
                level_score = 0

            # Simplified scoring (prioritize level for immediate response)
            combined_score = (
                0.5 * level_score  # Higher weight on signal strength for fast response
                + 0.2 * voice_ratio_score  # Reduced weight on complex analysis
                + 0.15 * centroid_score
                + 0.15 * quality_score
            )

            # Apply hysteresis
            threshold = (
                self.squelch.close_threshold
                if self.squelch.is_open
                else self.squelch.open_threshold
            )
            self.squelch.is_open = combined_score >= threshold

            stats = {
                "current_db": current_db,
                "noise_floor_db": noise_floor_db,
                "level_margin_db": level_margin_db,
                "voice_energy": voice_energy,
                "total_energy": total_energy,
                "spectral_centroid": spectral_centroid,
                "voice_quality": voice_quality,
                "level_score": level_score,
                "voice_ratio_score": voice_ratio_score,
                "centroid_score": centroid_score,
                "quality_score": quality_score,
                "combined_score": combined_score,
                "squelch_open": self.squelch.is_open,
            }

            return self.squelch.is_open, stats

        except Exception as e:
            self.logger.error(f"Error in immediate squelch processing: {e}")
            # Return safe defaults
            return False, {
                "current_db": -60.0,
                "noise_floor_db": -60.0,
                "level_margin_db": 0.0,
                "voice_energy": 0,
                "total_energy": 1,
                "spectral_centroid": 1000,
                "voice_quality": 0,
                "level_score": 0,
                "voice_ratio_score": 0,
                "centroid_score": 0,
                "quality_score": 0,
                "combined_score": 0,
                "squelch_open": False,
            }

    def _process_audio_chunk(self, audio_chunk: np.ndarray):
        """
        Process audio chunk using the threaded bus system.
        The @cpu_intensive decorator on squelch.process() handles threading automatically.
        """
        try:
            # Always add to pre-record buffer (unless we're already recording)
            if not self.recording:
                self._add_to_pre_record_buffer(audio_chunk)

            # The squelch.process method is marked with @cpu_intensive, so it will
            # automatically run in a separate thread via the bus system
            squelch_open, stats = self.squelch.process(audio_chunk)

            # Handle squelch state changes (fast operations, synchronous)
            current_time = time.time()

            if squelch_open and not self.recording:
                # Squelch opened - start recording immediately (no delay needed now)
                self.start_recording()
                self.squelch_open_time = None

            elif not squelch_open:
                # Squelch closed
                self.squelch_open_time = None

                if self.recording:
                    self.silence_counter += 1
                    chunks_for_silence = self.silence_duration * (
                        self.sample_rate / self.chunk_size
                    )

                    if self.silence_counter >= chunks_for_silence:
                        self.stop_recording()
            else:
                # Squelch is open and recording
                self.silence_counter = 0

            # Schedule statistics publishing asynchronously
            if (
                current_time - self.last_stats_publish_time >= self.stats_interval
                and self.event_loop
            ):
                # Use the stored event loop reference for thread-safe async operations
                asyncio.run_coroutine_threadsafe(
                    self._publish_enhanced_stats(stats), self.event_loop
                )
                self.last_stats_publish_time = current_time

        except Exception as e:
            self.logger.error(f"Error in audio chunk processing: {e}")

    def process_audio_stream(self):
        """Enhanced audio stream processing with advanced squelch using FFmpeg streaming."""
        self.logger.info("Starting enhanced audio stream processing with FFmpeg")

        max_retries = 3
        retry_delay = 5.0  # seconds

        for attempt in range(max_retries):
            if not self.enabled:
                self.logger.info("Audio processing disabled, stopping")
                return

            self.logger.info(
                f"Audio stream attempt {attempt + 1}/{max_retries} with device: {self.audio_device}"
            )

            if self._process_audio_stream_once():
                # Successful run, don't retry unless it fails again
                break
            else:
                # Failed, try to find a different device for next attempt
                if attempt < max_retries - 1:
                    self.logger.warning(
                        f"Audio stream failed, retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)

                    # Try to find a working device for next attempt
                    self._test_audio_device()
                else:
                    self.logger.error("All audio stream attempts failed")
        """Enhanced audio stream processing with advanced squelch using FFmpeg streaming."""
        self.logger.info("Starting enhanced audio stream processing with FFmpeg")

        max_retries = 3
        retry_delay = 5.0  # seconds

        for attempt in range(max_retries):
            if not self.enabled:
                self.logger.info("Audio processing disabled, stopping")
                return

            self.logger.info(
                f"Audio stream attempt {attempt + 1}/{max_retries} with device: {self.audio_device}"
            )

            if self._process_audio_stream_once():
                # Successful run, don't retry unless it fails again
                break
            else:
                # Failed, try to find a different device for next attempt
                if attempt < max_retries - 1:
                    self.logger.warning(
                        f"Audio stream failed, retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)

                    # Try to find a working device for next attempt
                    self._test_audio_device()
                else:
                    self.logger.error("All audio stream attempts failed")

    def _process_audio_stream_once(self) -> bool:
        """Single attempt at audio stream processing. Returns True if successful."""
        # Use FFmpeg for streaming with minimal latency settings
        ffmpeg_cmd = [
            "ffmpeg",
            "-f",
            "alsa",
            "-i",
            self.audio_device,
            "-ac",
            "1",  # Mono
            "-ar",
            str(self.sample_rate),  # Sample rate
            "-acodec",
            "pcm_s16le",  # Direct PCM encoding
            "-fflags",
            "nobuffer",  # Disable buffering for low latency
            "-flags",
            "low_delay",  # Enable low delay mode
            "-strict",
            "experimental",
            "-f",
            "s16le",  # 16-bit little-endian PCM
            "-",  # Output to stdout
        ]

        process = None
        try:
            # Start FFmpeg process for continuous audio streaming with minimal buffering
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # Completely unbuffered
            )

            if not process.stdout:
                self.logger.error("Failed to get FFmpeg stdout stream")
                return False

            self.logger.info(
                f"FFmpeg low-latency audio streaming started with device: {self.audio_device}"
            )

            # Use smaller chunks for immediate response
            bytes_per_sample = 2  # 16-bit = 2 bytes
            # Reduce chunk size for faster response (was self.chunk_size)
            small_chunk_size = min(
                256, self.chunk_size
            )  # Use 256 samples max for low latency
            chunk_bytes = small_chunk_size * bytes_per_sample

            # Track successful operation
            successful_chunks = 0
            min_successful_chunks = 10

            # Buffer for accumulating partial data
            audio_buffer = bytearray()

            try:
                while True:
                    if not self.enabled:
                        self.logger.info("Audio processing disabled, stopping stream")
                        return successful_chunks >= min_successful_chunks

                    try:
                        # Check if FFmpeg process is still running
                        if process.poll() is not None:
                            # Process has terminated, try to get error info
                            _, stderr = process.communicate()
                            error_msg = stderr.decode() if stderr else "Unknown error"
                            self.logger.error(f"FFmpeg process terminated: {error_msg}")
                            return successful_chunks >= min_successful_chunks

                        # Use non-blocking read with select for immediate response
                        import select

                        # Check if data is available for reading (timeout = 0.001s for immediate response)
                        ready, _, _ = select.select([process.stdout], [], [], 0.001)

                        if not ready:
                            # No data available, continue immediately (don't sleep)
                            continue

                        # Read available data (may be less than chunk_bytes)
                        raw_data = process.stdout.read(chunk_bytes)

                        if not raw_data:
                            # No data available, FFmpeg might have stopped
                            continue

                        # Add to buffer
                        audio_buffer.extend(raw_data)

                        # Process complete chunks immediately when available
                        while len(audio_buffer) >= chunk_bytes:
                            # Extract one complete chunk
                            chunk_data = audio_buffer[:chunk_bytes]
                            audio_buffer = audio_buffer[chunk_bytes:]

                            # Convert raw bytes to numpy array
                            audio_chunk = np.frombuffer(chunk_data, dtype=np.int16)

                            # Process audio chunk with immediate response
                            try:
                                self._process_audio_chunk_immediate(audio_chunk)
                                successful_chunks += 1
                            except Exception as processing_error:
                                # If processing fails, log but continue with audio stream
                                self.logger.debug(
                                    f"Audio processing error: {processing_error}"
                                )
                                successful_chunks += 1  # Still count as successful

                    except Exception as e:
                        self.logger.error(f"Error reading audio chunk: {e}")
                        # Don't sleep on errors, continue immediately

            finally:
                pass  # No async processing cleanup needed anymore

        except Exception as e:
            self.logger.error(f"Fatal error in audio stream processing: {e}")
            return False
        finally:
            # Clean up FFmpeg process
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.logger.warning("FFmpeg termination timeout, force killing")
                    process.kill()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.logger.error("Could not kill FFmpeg process")
                except Exception as e:
                    self.logger.error(f"Error terminating FFmpeg process: {e}")

            self.logger.info("Audio stream processing stopped")

    async def _publish_enhanced_stats(self, stats: dict):
        """Publish enhanced statistics including spectral analysis and performance metrics."""
        await self.bus.publish(
            RxNoiseFloorStatsEvent(
                ambient_db=float(stats["noise_floor_db"]),
                current_db=float(stats["current_db"]),
                delta_db=float(stats["level_margin_db"]),
            )
        )

        # Log detailed stats for debugging/monitoring including performance
        performance_info = ""
        if "avg_process_time_ms" in stats and "max_process_time_ms" in stats:
            performance_info = f", Avg process: {stats['avg_process_time_ms']:.2f}ms, Max: {stats['max_process_time_ms']:.2f}ms"

        self.logger.info(
            f"Audio Stats - "
            f"Level: {stats['current_db']:.1f}dB, "
            f"Noise: {stats['noise_floor_db']:.1f}dB, "
            f"Margin: {stats['level_margin_db']:.1f}dB, "
            f"Squelch: {'OPEN' if stats['squelch_open'] else 'CLOSED'}, "
            f"Voice ratio: {stats['voice_ratio_score']:.2f}, "
            f"Voice quality: {stats['voice_quality']:.2f}, "
            f"Spectral centroid: {stats['spectral_centroid']:.0f}Hz"
            f"{performance_info}"
        )

    async def start(self):
        """Start the enhanced audio processing worker."""
        self.logger.info("Enhanced RxListenWorker starting...")

        # Start the main audio stream processing in a separate thread
        # This allows the sync FFmpeg reading loop to run without blocking the async loop
        asyncio.create_task(asyncio.to_thread(self.process_audio_stream))

    async def stop(self):
        """Stop the audio processing worker and clean up resources."""
        self.logger.info("Stopping Enhanced RxListenWorker...")

        self.enabled = False

        # Stop any ongoing recording
        if self.recording:
            self.stop_recording()

        self.logger.info("Enhanced RxListenWorker stopped")


class RXListenPrintStatsWorker(Worker):
    def __init__(self, bus: MessageBus):
        super().__init__(bus)
        self.bus.subscribe(RxNoiseFloorStatsEvent, self.handle_noise_floor_stats)

    async def handle_noise_floor_stats(self, event: RxNoiseFloorStatsEvent):
        # Print or log the noise floor statistics.
        self.logger.info(
            f"Ambient: {event.ambient_db:.2f} dB, Current: {event.current_db:.2f} dB, Delta: {event.delta_db:.2f} dB"
        )

    async def start(self):
        while True:
            await asyncio.sleep(1)
