import asyncio
import datetime
import os
import subprocess
import time
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
from bus import MessageBus, Worker
from events import (
    RxNoiseFloorStatsEvent,
    RxRecordingCompleteEvent,
    RxRecordingEndedEvent,
    RxRecordingStartedEvent,
    TelegramVoiceMessageDownloadedEvent,
    TxMessagePlaybackEndedEvent,
    TxMessagePlaybackStartedEvent,
)
from scipy import signal
from scipy.io import wavfile


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

                # Penalize pure tones (very narrow spectrum)
                spectral_spread = (
                    np.sum(
                        (frequencies[: len(magnitude)] - spectral_centroid) ** 2
                        * magnitude**2
                    )
                    / total_energy
                )
                spread_factor = min(1.0, spectral_spread / 1000000)  # Normalize spread
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
        self.level_threshold_db = 8.0  # dB above noise floor (balanced sensitivity)
        self.voice_energy_ratio = (
            0.15  # Min ratio of voice energy to total (relaxed for voice)
        )
        self.spectral_centroid_min = 200  # Hz (wide range for various voices)
        self.spectral_centroid_max = 2500  # Hz (wide range for various voices)
        self.min_voice_quality = (
            0.02  # Minimum voice quality score (moderate requirement)
        )
        self.min_absolute_level = -50.0  # Absolute minimum level in dB (sensitive)

        # Hysteresis for stability - optimized for voice detection
        self.open_threshold = 0.75  # Reduced for better voice sensitivity
        self.close_threshold = 0.60  # Maintain gap for stability
        self.is_open = False

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

        # Voice detection thresholds (tuned based on good voice vs noise samples)
        self.min_voice_energy_ratio = 0.10  # Minimum voice band energy ratio (lowered)
        self.min_voice_quality_score = 0.02  # Minimum voice quality (lowered)
        self.min_spectral_centroid = 300  # Hz - minimum for voice (lowered)
        self.max_spectral_centroid = 2500  # Hz - maximum for voice (raised)
        self.min_analysis_duration = 0.5  # Minimum seconds to analyze
        self.voice_consistency_threshold = (
            0.35  # Portion of chunks that must pass voice tests (lowered)
        )

        # Energy and dynamics thresholds
        self.min_energy_db = -60.0  # Minimum energy level (lowered for weak signals)
        self.min_dynamic_range_db = 6.0  # Minimum dynamic range for voice (lowered)

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
                max_energy_cv_for_voice = (
                    2.0  # Threshold for sustained vs impulsive energy
                )
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
                0.4  # At least 40% of chunks should have sustained energy
            )

            # Enhanced voice detection criteria
            # Voice should have some spectral variability (not completely stable like pure noise)
            min_spectral_variability = (
                0.01  # Minimum variability to distinguish from pure tones/static
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
                and spectral_stability
                > min_spectral_variability  # Reject pure tones/static
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


class EnhancedTxPlayWorker(Worker):
    """Enhanced transmitter worker with advanced audio processing."""

    def __init__(self, bus: MessageBus):
        super().__init__(bus)

        # Event subscriptions
        self.bus.subscribe(TelegramVoiceMessageDownloadedEvent, self.queue_event)
        self.bus.subscribe(RxRecordingStartedEvent, self.on_recording_started)
        self.bus.subscribe(RxRecordingEndedEvent, self.on_recording_finished)

        # Playback configuration
        self.enabled = True
        self.timeout_seconds = 30
        self.wake_tone_frequency = 1750  # Hz - better for radio transmission
        self.wake_tone_duration = 0.3  # seconds
        self.post_tx_delay = 2.5  # seconds for radio to return to RX mode

        # Audio processing parameters
        self.target_lufs = -16  # Loudness target
        self.limiter_ceiling = -1  # dB - prevent clipping
        self.compressor_ratio = 3.0  # Compression ratio
        self.noise_reduction_amount = 8  # dB of noise reduction

    async def on_recording_started(self, event: RxRecordingStartedEvent):
        """Disable playback when recording starts to prevent interference."""
        self.enabled = False
        self.logger.debug("TX disabled - RX recording started")

    async def on_recording_finished(self, event: RxRecordingEndedEvent):
        """Enable playback when recording ends."""
        self.enabled = True
        self.logger.debug("TX enabled - RX recording ended")

    async def handle_event(self, event: TelegramVoiceMessageDownloadedEvent):
        """Handle incoming voice message for transmission."""
        if not self.enabled:
            self.logger.info(f"TX disabled, queuing message: {event.filepath}")
            # Re-queue the event to try later
            await asyncio.sleep(1.0)
            await self.queue_event(event)
            return

        self.logger.info(f"Processing voice message for TX: {event.filepath}")
        await self.bus.publish(TxMessagePlaybackStartedEvent())

        try:
            await self.play_enhanced_audio(event.filepath)
        except Exception as e:
            self.logger.error(f"Error during enhanced playback: {e}")
        finally:
            await asyncio.sleep(self.post_tx_delay)
            await self.bus.publish(TxMessagePlaybackEndedEvent())

    async def play_enhanced_audio(self, filepath: str):
        """Play audio with enhanced processing optimized for radio transmission."""

        # Step 1: Pre-process the audio for radio transmission
        processed_filepath = await self._preprocess_for_radio(filepath)

        if not processed_filepath:
            self.logger.error("Failed to preprocess audio")
            return

        try:
            # Step 2: Generate wake tone to activate radio TX
            await self._play_wake_tone()

            # Step 3: Small delay for radio to fully key up
            await asyncio.sleep(0.2)

            # Step 4: Play the processed audio
            await self._play_audio_file(processed_filepath)

        finally:
            # Clean up processed file
            if os.path.exists(processed_filepath):
                os.remove(processed_filepath)
            # Clean up original file
            if os.path.exists(filepath):
                os.remove(filepath)

    async def _preprocess_for_radio(self, input_filepath: str) -> Optional[str]:
        """
        Apply comprehensive audio processing optimized for radio transmission.

        Returns path to processed file, or None if processing failed.
        """
        try:
            processed_filepath = input_filepath.replace(".ogg", "_radio_processed.ogg")

            # Comprehensive filter chain for radio transmission
            filter_complex = (
                "[0:a]"
                # 1. Normalize input level
                "loudnorm=I=-23:LRA=7:TP=-2:offset=0,"
                # 2. High-pass filter to remove rumble and handling noise
                "highpass=f=200:poles=2,"
                # 3. Pre-emphasis for radio transmission (boost highs)
                "treble=g=3:f=1000:width_type=h:width=1000,"
                # 4. Bandpass filter optimized for radio voice
                "lowpass=f=3000:poles=2,"
                # 5. Dynamic range compression for consistent levels
                "compand=attacks=0.003:decays=0.1:points=-80/-80|-43/-43|-30/-25|-18/-15:soft-knee=6:gain=0:volume=-90,"
                # 6. Gentle noise reduction
                f"afftdn=nr={self.noise_reduction_amount}:nf=-40:tn=1,"
                # 7. Peak limiter to prevent overmodulation
                f"alimiter=level_in=1:level_out=1:limit={self.limiter_ceiling}dB:attack=3:release=20,"
                # 8. Final loudness normalization for consistent TX levels
                f"loudnorm=I={self.target_lufs}:LRA=5:TP=-1[a]"
            )

            command = [
                "ffmpeg",
                "-y",
                "-i",
                input_filepath,
                "-filter_complex",
                filter_complex,
                "-map",
                "[a]",
                "-ar",
                "48000",
                "-c:a",
                "libopus",
                "-b:a",
                "96k",  # Higher quality for TX
                processed_filepath,
            ]

            self.logger.debug(f"Preprocessing audio with command: {' '.join(command)}")

            result = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )

            _, stderr = await result.communicate()

            if result.returncode == 0:
                self.logger.debug("Audio preprocessing completed successfully")
                return processed_filepath
            else:
                self.logger.error(f"FFmpeg preprocessing failed: {stderr.decode()}")
                return None

        except Exception as e:
            self.logger.error(f"Error in audio preprocessing: {e}")
            return None

    async def _play_wake_tone(self):
        """Play a wake-up tone to activate radio PTT reliably."""
        process = None
        try:
            with open("/dev/null", "w") as devnull:
                # Generate and play wake tone
                process = await asyncio.create_subprocess_exec(
                    "ffplay",
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    "-f",
                    "lavfi",
                    "-i",
                    f"sine=frequency={self.wake_tone_frequency}:duration={self.wake_tone_duration}",
                    stdout=devnull,
                    stderr=devnull,
                )

                await asyncio.wait_for(
                    process.wait(), timeout=self.wake_tone_duration + 1.0
                )
                self.logger.debug(
                    f"Wake tone played: {self.wake_tone_frequency}Hz for {self.wake_tone_duration}s"
                )

        except asyncio.TimeoutError:
            self.logger.warning("Wake tone playback timeout")
            if process:
                try:
                    process.terminate()
                    await process.wait()
                except:
                    pass
        except Exception as e:
            self.logger.error(f"Error playing wake tone: {e}")

    async def _play_audio_file(self, filepath: str):
        """Play processed audio file with timeout protection."""
        try:
            with open("/dev/null", "w") as devnull:
                process = await asyncio.create_subprocess_exec(
                    "ffplay",
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    "-af",
                    "volume=0.8",  # Slight volume reduction for safety
                    filepath,
                    stdout=devnull,
                    stderr=devnull,
                )

                try:
                    await asyncio.wait_for(process.wait(), timeout=self.timeout_seconds)
                    self.logger.debug(f"Audio playback completed: {filepath}")
                except asyncio.TimeoutError:
                    self.logger.warning(f"Playback timeout for {filepath}, terminating")
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        self.logger.warning("Force killing playback process")
                        process.kill()
                        await process.wait()

        except Exception as e:
            self.logger.error(f"Error playing audio file {filepath}: {e}")

    async def start(self):
        """Start processing the queue."""
        asyncio.create_task(self.process_queue())


class EnhancedRxListenWorker(Worker):
    """Enhanced receiver worker with advanced audio processing."""

    def __init__(self, bus: MessageBus, sample_rate=48000, chunk_size=512):
        super().__init__(bus)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        # Recording parameters
        self.silence_duration = 2.0  # Seconds to wait before stopping recording
        self.minimal_recording_duration = 1.0  # Min duration after trimming
        self.enabled = True
        self.recording = False
        self.process = None
        self.silence_counter = 0

        # File paths
        self.recording_filepath: Optional[str] = None
        self.recording_start_time: Optional[datetime.datetime] = None

        # Advanced audio processing
        self.squelch = AdvancedSquelch(sample_rate)
        self.voice_detector = VoiceDetector(sample_rate)

        # Statistics and monitoring
        self.stats_interval = 5.0  # Publish stats every 5 seconds
        self.last_stats_publish_time = 0

        # Squelch state tracking
        self.squelch_open_time: Optional[float] = None
        self.min_squelch_open_duration = 0.5  # Minimum time squelch must be open

        # Event subscriptions
        self.bus.subscribe(TxMessagePlaybackStartedEvent, self.on_playback_started)
        self.bus.subscribe(TxMessagePlaybackEndedEvent, self.on_playback_finished)

    async def on_playback_started(self, event: TxMessagePlaybackStartedEvent):
        """Disable listening during playback to prevent feedback."""
        self.enabled = False
        if self.recording:
            self.logger.info("Stopping recording due to TX playback starting")
            self.stop_recording()

    async def on_playback_finished(self, event: TxMessagePlaybackEndedEvent):
        """Re-enable listening after playback ends."""
        self.enabled = True

    def start_recording(self):
        """Start recording with enhanced processing."""
        if self.recording:
            return

        os.makedirs("recordings", exist_ok=True)
        timestamp = datetime.datetime.now().timestamp()
        filename = f"recording_{timestamp:.3f}.wav"  # Use WAV for better processing
        self.recording_filepath = os.path.join("recordings", filename)
        self.recording_start_time = datetime.datetime.now()

        # Record raw audio first, then process
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "alsa",
            "-i",
            "default",
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-c:a",
            "pcm_s16le",  # Raw PCM for processing
            self.recording_filepath,
        ]

        try:
            self.process = subprocess.Popen(
                command, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
            )
            self.recording = True
            self.silence_counter = 0
            asyncio.run(self.bus.publish(RxRecordingStartedEvent()))
            self.logger.info(f"Enhanced recording started: {self.recording_filepath}")
        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            self.recording = False

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
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            finally:
                self.process = None

        self.recording = False
        asyncio.run(self.bus.publish(RxRecordingEndedEvent()))

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

    def _process_recording(self, duration: float):
        """Apply advanced processing to recorded audio."""
        if not self.recording_filepath:
            return

        try:
            processed_filepath = self.recording_filepath.replace(
                ".wav", "_processed.ogg"
            )

            # Advanced FFmpeg filter chain for radio processing
            filter_complex = (
                # 1. Trim to remove trailing silence
                f"[0:a]atrim=start=0:duration={duration},"
                # 2. High-pass filter to remove low-frequency noise
                "highpass=f=300,"
                # 3. De-emphasis (radio typically has pre-emphasis)
                "treble=g=-6:f=1000:width_type=h:width=1000,"
                # 4. Bandpass filter for voice clarity
                "lowpass=f=3400,"
                # 5. Dynamic range compression for consistent levels
                "compand=attacks=0.01:decays=0.5:points=-70/-70|-60/-50|-30/-20|-10/-10:soft-knee=6:gain=0:volume=-20,"
                # 6. Noise reduction (gentle)
                "afftdn=nr=10:nf=-50:tn=1,"
                # 7. Final limiter to prevent clipping
                "alimiter=level_in=1:level_out=1:limit=-1dB:attack=5:release=50[a]"
            )

            command = [
                "ffmpeg",
                "-y",
                "-i",
                self.recording_filepath,
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
            # Clean up raw recording file
            if os.path.exists(self.recording_filepath):
                os.remove(self.recording_filepath)

    def process_audio_stream(self):
        """Enhanced audio stream processing with advanced squelch."""
        self.logger.info("Starting enhanced audio stream processing")

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.chunk_size,
            ) as stream:

                while True:
                    if not self.enabled:
                        time.sleep(0.01)
                        continue

                    try:
                        data, overflowed = stream.read(self.chunk_size)

                        if overflowed:
                            self.logger.warning("Audio input overflow detected")

                        # Flatten the data array
                        audio_chunk = data.flatten()

                        # Apply advanced squelch processing
                        squelch_open, stats = self.squelch.process(audio_chunk)

                        # Handle squelch state changes
                        current_time = time.time()

                        if squelch_open and not self.recording:
                            # Squelch opened
                            if self.squelch_open_time is None:
                                self.squelch_open_time = current_time
                            elif (
                                current_time - self.squelch_open_time
                            ) > self.min_squelch_open_duration:
                                # Squelch has been open long enough to start recording
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

                        # Publish statistics periodically
                        if (
                            current_time - self.last_stats_publish_time
                            >= self.stats_interval
                        ):
                            asyncio.run(self._publish_enhanced_stats(stats))
                            self.last_stats_publish_time = current_time

                    except Exception as e:
                        self.logger.error(f"Error processing audio chunk: {e}")
                        time.sleep(0.01)

        except Exception as e:
            self.logger.error(f"Fatal error in audio stream processing: {e}")

    async def _publish_enhanced_stats(self, stats: dict):
        """Publish enhanced statistics including spectral analysis."""
        await self.bus.publish(
            RxNoiseFloorStatsEvent(
                ambient_db=float(stats["noise_floor_db"]),
                current_db=float(stats["current_db"]),
                delta_db=float(stats["level_margin_db"]),
            )
        )

        # Log detailed stats for debugging/monitoring
        self.logger.info(
            f"Audio Stats - "
            f"Level: {stats['current_db']:.1f}dB, "
            f"Noise: {stats['noise_floor_db']:.1f}dB, "
            f"Margin: {stats['level_margin_db']:.1f}dB, "
            f"Squelch: {'OPEN' if stats['squelch_open'] else 'CLOSED'}, "
            f"Voice ratio: {stats['voice_ratio_score']:.2f}, "
            f"Voice quality: {stats['voice_quality']:.2f}, "
            f"Spectral centroid: {stats['spectral_centroid']:.0f}Hz"
        )

    async def start(self):
        """Start the enhanced audio processing worker."""
        self.logger.info("Enhanced RxListenWorker starting...")
        asyncio.create_task(asyncio.to_thread(self.process_audio_stream))


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
