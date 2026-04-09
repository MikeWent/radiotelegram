"""Behavior tests: every audio file in tests/noise/ must be rejected as non-voice.

These are integration tests that run the real TEN-VAD model against recorded
noise samples. They require ten_vad, scipy, and ffmpeg to be available
(i.e. run inside Docker).
"""

import glob
import os

import pytest

NOISE_DIR = os.path.join(os.path.dirname(__file__), "noise")
NOISE_FILES = sorted(glob.glob(os.path.join(NOISE_DIR, "*")))

ten_vad = pytest.importorskip("ten_vad", reason="ten_vad not installed (run in Docker)")

from radiotelegram.voice_detection import VoiceDetector


@pytest.fixture(scope="module")
def detector():
    return VoiceDetector()


@pytest.mark.parametrize(
    "filepath",
    NOISE_FILES,
    ids=[os.path.basename(f) for f in NOISE_FILES],
)
def test_noise_file_rejected(detector, filepath):
    is_voice, analysis = detector.analyze_recording(filepath)

    g = analysis.get
    detail = (
        f"voice_ratio={g('voice_ratio', 0):.3f} "
        f"(min={g('min_voice_ratio', 0)}), "
        f"max_prob={g('max_probability', 0):.3f} "
        f"(min={g('min_max_probability', 0)}), "
        f"avg={g('avg_probability', 0):.3f}, "
        f"p90={g('p90', 0):.3f}, "
        f"p95={g('p95', 0):.3f}, "
        f"frames={g('voice_frames', 0)}/{g('total_frames', 0)}"
    )
    assert (
        is_voice is False
    ), f"Noise file classified as voice: {os.path.basename(filepath)}\n{detail}"
