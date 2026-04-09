"""Tests for radiotelegram.voice_detection — recording-level voice classification.

Verifies:
  - Recordings too short are rejected before VAD runs
  - Silence (all zeros) is never classified as voice
  - ffmpeg conversion failure returns a clean error
  - Temp file is always cleaned up after reading
  - Stereo input is reduced to mono without crashing
  - Generic exceptions are caught and surfaced as error dicts
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _mock_ten_vad():
    with patch("radiotelegram.voice_detection.TenVad") as mock_cls:
        vad = MagicMock()
        vad.process.return_value = (0.0, 0)
        mock_cls.return_value = vad
        yield mock_cls


from radiotelegram.voice_detection import VoiceDetector


@pytest.fixture
def detector():
    return VoiceDetector(
        sample_rate=48000,
        hop_size=256,
        threshold=0.5,
        min_voice_ratio=0.05,
        min_max_probability=0.75,
        min_analysis_duration=0.5,
    )


class TestVoiceDetectionDecisions:
    @patch("radiotelegram.voice_detection.subprocess.run")
    def test_ffmpeg_failure_rejects_cleanly(self, mock_run, detector):
        mock_run.return_value = MagicMock(returncode=1, stderr="codec error")
        is_voice, info = detector.analyze_recording("/tmp/broken.wav")
        assert is_voice is False
        assert "error" in info

    @patch("radiotelegram.voice_detection.os.remove")
    @patch("radiotelegram.voice_detection.wavfile.read")
    @patch("radiotelegram.voice_detection.subprocess.run")
    def test_too_short_recording_rejected(self, mock_run, mock_wav, mock_rm, detector):
        mock_run.return_value = MagicMock(returncode=0)
        # 0.1 s of audio at 16 kHz — well below the 0.5 s minimum
        mock_wav.return_value = (16000, np.zeros(1600, dtype=np.int16))

        is_voice, info = detector.analyze_recording("/tmp/tiny.wav")
        assert is_voice is False
        assert "too short" in info.get("error", "").lower()

    @patch("radiotelegram.voice_detection.os.remove")
    @patch("radiotelegram.voice_detection.wavfile.read")
    @patch("radiotelegram.voice_detection.subprocess.run")
    def test_silence_is_not_classified_as_voice(
        self, mock_run, mock_wav, mock_rm, detector
    ):
        mock_run.return_value = MagicMock(returncode=0)
        # 1 s of digital silence at 16 kHz
        mock_wav.return_value = (16000, np.zeros(16000, dtype=np.int16))

        is_voice, info = detector.analyze_recording("/tmp/silence.wav")
        assert is_voice is False
        assert (
            info["voice_ratio"] == 0.0
            or info["max_probability"] <= detector.min_max_probability
        )

    @patch("radiotelegram.voice_detection.os.remove")
    @patch("radiotelegram.voice_detection.wavfile.read")
    @patch("radiotelegram.voice_detection.subprocess.run")
    def test_stereo_input_reduced_to_mono(self, mock_run, mock_wav, mock_rm, detector):
        mock_run.return_value = MagicMock(returncode=0)
        stereo = np.zeros((16000, 2), dtype=np.int16)
        mock_wav.return_value = (16000, stereo)

        # Must not crash — stereo is handled by taking first channel
        is_voice, info = detector.analyze_recording("/tmp/stereo.wav")
        assert isinstance(is_voice, bool)
        assert "error" not in info  # no error, analysis ran

    @patch("radiotelegram.voice_detection.os.remove")
    @patch("radiotelegram.voice_detection.wavfile.read")
    @patch("radiotelegram.voice_detection.subprocess.run")
    def test_temp_file_always_cleaned_up(self, mock_run, mock_wav, mock_rm, detector):
        mock_run.return_value = MagicMock(returncode=0)
        mock_wav.return_value = (16000, np.zeros(16000, dtype=np.int16))

        detector.analyze_recording("/tmp/test.wav")
        mock_rm.assert_called_once_with("/tmp/test.wav_vad.wav")

    @patch("radiotelegram.voice_detection.subprocess.run")
    def test_generic_exception_caught(self, mock_run, detector):
        mock_run.side_effect = PermissionError("no access")
        is_voice, info = detector.analyze_recording("/tmp/locked.wav")
        assert is_voice is False
        assert "error" in info

    @patch("radiotelegram.voice_detection.os.remove")
    @patch("radiotelegram.voice_detection.wavfile.read")
    @patch("radiotelegram.voice_detection.subprocess.run")
    def test_analysis_dict_contains_expected_keys(
        self, mock_run, mock_wav, mock_rm, detector
    ):
        mock_run.return_value = MagicMock(returncode=0)
        mock_wav.return_value = (16000, np.zeros(16000, dtype=np.int16))

        _, info = detector.analyze_recording("/tmp/x.wav")
        for key in (
            "duration",
            "voice_ratio",
            "avg_probability",
            "max_probability",
            "voice_frames",
            "total_frames",
            "is_voice",
        ):
            assert key in info
