"""Voice detection functionality for analyzing recorded audio."""

import os
import subprocess
from typing import Tuple

import numpy as np
from scipy.io import wavfile

from radiotelegram.audio_analysis import SpectralAnalyzer
from radiotelegram.bus import cpu_intensive


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
