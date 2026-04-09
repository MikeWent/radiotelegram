"""Tests for radiotelegram.audio_analysis — noise floor and squelch behaviour."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from radiotelegram.audio_analysis import AdaptiveNoiseFloor


@pytest.fixture(autouse=True)
def _mock_ten_vad():
    with patch("radiotelegram.audio_analysis.TenVad") as mock_cls:
        vad = MagicMock()
        vad.process.return_value = (0.0, 0)
        mock_cls.return_value = vad
        yield mock_cls


from radiotelegram.audio_analysis import AdvancedSquelch


# ── AdaptiveNoiseFloor ───────────────────────────────────────────


class TestNoiseFloorAdaptation:
    def test_converges_toward_steady_noise_level(self):
        nf = AdaptiveNoiseFloor()
        for _ in range(2000):
            nf.update(-45.0, is_signal_present=False)
        result = nf.update(-45.0, is_signal_present=False)
        assert abs(result - (-45.0)) < 0.5

    def test_freezes_during_signal(self):
        nf = AdaptiveNoiseFloor()
        for _ in range(500):
            nf.update(-50.0, is_signal_present=False)
        before = nf.update(-50.0, is_signal_present=False)
        for _ in range(100):
            nf.update(-10.0, is_signal_present=True)
        after = nf.update(-50.0, is_signal_present=False)
        assert abs(after - before) < 1.0

    def test_clamped_below_minimum(self):
        nf = AdaptiveNoiseFloor()
        for _ in range(5000):
            nf.update(-200.0, is_signal_present=False)
        assert nf.fast_noise_floor >= nf.min_noise_floor
        assert nf.slow_noise_floor >= nf.min_noise_floor

    def test_clamped_above_maximum(self):
        nf = AdaptiveNoiseFloor()
        for _ in range(5000):
            nf.update(0.0, is_signal_present=False)
        assert nf.fast_noise_floor <= nf.max_noise_floor
        assert nf.slow_noise_floor <= nf.max_noise_floor

    def test_fast_tracks_quicker_than_slow(self):
        nf = AdaptiveNoiseFloor(fast_alpha=0.5, slow_alpha=0.01)
        nf.update(-40.0, is_signal_present=False)
        assert nf.fast_noise_floor > nf.slow_noise_floor


# ── AdvancedSquelch ──────────────────────────────────────────────


class TestSquelchBehaviour:
    def setup_method(self):
        self.squelch = AdvancedSquelch(sample_rate=48000)

    def _settle_noise_floor(self, n=200):
        silence = np.zeros(512, dtype=np.int16)
        for _ in range(n):
            self.squelch.process(silence)

    def test_stays_closed_on_silence(self):
        silence = np.zeros(512, dtype=np.int16)
        is_open, _ = self.squelch.process(silence)
        assert is_open is False

    def test_very_quiet_signal_stays_closed(self):
        whisper = np.ones(512, dtype=np.int16)
        _, stats = self.squelch.process(whisper)
        assert stats["level_score"] == 0

    def test_hysteresis_requires_higher_score_to_open_than_to_close(self):
        assert self.squelch.open_threshold > self.squelch.close_threshold

    def test_hysteresis_gap_is_meaningful(self):
        gap = self.squelch.open_threshold - self.squelch.close_threshold
        assert gap >= 0.05

    def test_returns_safe_defaults_on_error(self):
        with patch.object(self.squelch.noise_floor, "update", side_effect=RuntimeError):
            is_open, stats = self.squelch.process(np.zeros(256, dtype=np.int16))
        assert is_open is False
        assert stats["squelch_open"] is False

    def test_stats_contain_expected_keys(self):
        _, stats = self.squelch.process(np.zeros(512, dtype=np.int16))
        for key in [
            "current_db", "noise_floor_db", "level_margin_db",
            "level_score", "vad_probability", "vad_flag",
            "combined_score", "squelch_open",
        ]:
            assert key in stats
