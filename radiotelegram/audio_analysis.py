"""Audio analysis: adaptive noise floor and squelch with TEN-VAD."""

import numpy as np
from ten_vad import TenVad


class AdaptiveNoiseFloor:
    """Tracks noise floor with fast and slow exponential averages."""

    def __init__(self, fast_alpha=0.1, slow_alpha=0.01):
        self.fast_alpha = fast_alpha
        self.slow_alpha = slow_alpha
        self.fast_noise_floor = -60.0
        self.slow_noise_floor = -60.0
        self.min_noise_floor = -80.0
        self.max_noise_floor = -30.0

    def update(self, current_db, is_signal_present=False):
        if not is_signal_present:
            self.fast_noise_floor = (
                self.fast_alpha * current_db
                + (1 - self.fast_alpha) * self.fast_noise_floor
            )
            self.slow_noise_floor = (
                self.slow_alpha * current_db
                + (1 - self.slow_alpha) * self.slow_noise_floor
            )
        self.fast_noise_floor = np.clip(
            self.fast_noise_floor, self.min_noise_floor, self.max_noise_floor
        )
        self.slow_noise_floor = np.clip(
            self.slow_noise_floor, self.min_noise_floor, self.max_noise_floor
        )
        return (self.fast_noise_floor + self.slow_noise_floor) / 2


class AdvancedSquelch:
    """Level + TEN-VAD squelch with hysteresis."""

    def __init__(self, sample_rate=48000, vad_hop_size=256):
        self.sample_rate = sample_rate
        self.noise_floor = AdaptiveNoiseFloor()
        self.open_threshold = 0.65
        self.close_threshold = 0.55
        self.min_absolute_level = -40.0
        self.is_open = False

        # Real-time TEN-VAD (16 kHz internally)
        self._vad = TenVad(hop_size=vad_hop_size, threshold=0.5)
        self._vad_ratio = sample_rate // 16000
        self._vad_hop_samples = vad_hop_size * self._vad_ratio
        self._vad_buf = np.empty(0, dtype=np.int16)
        self.vad_probability = 0.0
        self.vad_flag = 0

    def process(self, audio_chunk):
        """Process audio chunk. Returns (is_open, stats_dict)."""
        try:
            rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
            current_db = 20 * np.log10(rms / 32767.0 + 1e-6)
            noise_floor_db = self.noise_floor.update(current_db, self.is_open)
            level_margin_db = current_db - noise_floor_db

            # Feed TEN-VAD with downsampled audio
            self._vad_buf = np.append(self._vad_buf, audio_chunk)
            while len(self._vad_buf) >= self._vad_hop_samples:
                frame = self._vad_buf[: self._vad_hop_samples : self._vad_ratio].astype(
                    np.int16
                )
                self.vad_probability, self.vad_flag = self._vad.process(frame)
                self._vad_buf = self._vad_buf[self._vad_hop_samples :]

            level_score = max(0, min(1, (level_margin_db - 3) / 15))
            if current_db < self.min_absolute_level:
                level_score = 0

            combined = 0.5 * level_score + 0.5 * self.vad_probability
            threshold = self.close_threshold if self.is_open else self.open_threshold
            self.is_open = combined >= threshold

            return self.is_open, {
                "current_db": current_db,
                "noise_floor_db": noise_floor_db,
                "level_margin_db": level_margin_db,
                "level_score": level_score,
                "vad_probability": self.vad_probability,
                "vad_flag": self.vad_flag,
                "combined_score": combined,
                "squelch_open": self.is_open,
            }
        except Exception:
            return False, {
                "current_db": -60.0,
                "noise_floor_db": -60.0,
                "level_margin_db": 0.0,
                "level_score": 0,
                "vad_probability": 0.0,
                "vad_flag": 0,
                "combined_score": 0,
                "squelch_open": False,
            }
