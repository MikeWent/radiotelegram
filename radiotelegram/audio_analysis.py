"""Audio analysis components for radio reception."""

from typing import Tuple

import numpy as np

from radiotelegram.bus import cpu_intensive


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
