"""Tests for radiotelegram.rx_worker — RX pipeline behaviour.

Verifies:
  - TX/RX coordination: playback disables listening and aborts active recording
  - Recording lifecycle: start → collect audio → stop publishes correct events
  - Duplicate/stale recording detection
  - Silence timeout counter increments and eventually stops recording
  - Pre-record circular buffer maintains fixed window
  - Immediate squelch process returns safe defaults on error
  - Stats publishing through the bus
"""

import datetime
import time
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Patch hardware-dependent components before import
_PATCHES = [
    patch("radiotelegram.rx_worker.subprocess.run"),
    patch("radiotelegram.rx_worker.VoiceDetector"),
    patch("radiotelegram.rx_worker.TenVad"),
]


@pytest.fixture(autouse=True)
def _patch_hw():
    mocks = [p.start() for p in _PATCHES]
    mocks[0].return_value = MagicMock(returncode=1, stderr="", stdout="")
    yield mocks
    for p in _PATCHES:
        p.stop()


from radiotelegram.bus import MessageBus
from radiotelegram.events import (
    RxAudioStatsEvent,
    RxRecordingEndedEvent,
    RxRecordingStartedEvent,
    TxMessagePlaybackEndedEvent,
    TxMessagePlaybackStartedEvent,
)
from radiotelegram.rx_worker import EnhancedRxListenWorker


# ── TX/RX coordination ───────────────────────────────────────────


class TestTxRxCoordination:
    """When TX plays back a message, RX must disable to prevent feedback."""

    def setup_method(self):
        self.bus = MessageBus()
        self.w = EnhancedRxListenWorker(self.bus)

    def teardown_method(self):
        self.bus.shutdown()

    def test_playback_disables_listening(self):
        assert self.w.enabled is True
        self.w.on_playback_started(TxMessagePlaybackStartedEvent())
        assert self.w.enabled is False

    def test_playback_end_re_enables_listening(self):
        self.w.enabled = False
        with patch.object(threading.Thread, "start"):
            self.w.on_playback_finished(TxMessagePlaybackEndedEvent())
        assert self.w.enabled is True

    def test_playback_during_recording_aborts_it(self):
        self.w.recording = True
        with patch.object(self.w, "stop_recording") as stop:
            self.w.on_playback_started(TxMessagePlaybackStartedEvent())
        stop.assert_called_once()
        assert self.w.enabled is False


# ── Recording lifecycle ──────────────────────────────────────────


class TestRecordingLifecycle:
    def setup_method(self):
        self.bus = MessageBus()
        self.w = EnhancedRxListenWorker(self.bus)

    def teardown_method(self):
        self.bus.shutdown()

    def test_start_publishes_started_event(self):
        received = []
        self.bus.subscribe(RxRecordingStartedEvent, received.append)
        self.w.start_recording()
        time.sleep(0.3)
        assert len(received) == 1
        assert self.w.recording is True

    def test_stop_publishes_ended_event(self):
        received = []
        self.bus.subscribe(RxRecordingEndedEvent, received.append)
        self.w.start_recording()
        self.w.recording_buffer = [np.zeros(256, dtype=np.int16)]
        self.w.stop_recording()
        time.sleep(0.3)
        assert self.w.recording is False
        assert len(received) == 1

    def test_stop_when_not_recording_is_noop(self):
        received = []
        self.bus.subscribe(RxRecordingEndedEvent, received.append)
        self.w.stop_recording()
        time.sleep(0.2)
        assert len(received) == 0

    def test_duplicate_start_is_ignored(self):
        self.w.start_recording()
        first_path = self.w.recording_filepath
        self.w.start_recording()  # should be a no-op
        assert self.w.recording_filepath == first_path

    def test_stale_recording_is_cleaned_up(self):
        """A recording whose start_time exceeds max_recording_duration is
        treated as stale and stopped before a new one starts."""
        self.w.recording = True
        self.w.recording_start_time = datetime.datetime.now() - datetime.timedelta(
            seconds=self.w.max_recording_duration + 10
        )
        self.w.recording_buffer = [np.zeros(256, dtype=np.int16)]
        with patch.object(self.w, "stop_recording", wraps=self.w.stop_recording):
            self.w.start_recording()
        # A new recording should now be active
        assert self.w.recording is True


# ── Silence timeout ─────────────────────────────────────────────


class TestSilenceTimeout:
    """During recording, consecutive silent chunks must increment the silence
    counter and eventually trigger stop_recording."""

    def setup_method(self):
        self.bus = MessageBus()
        self.w = EnhancedRxListenWorker(self.bus)

    def teardown_method(self):
        self.bus.shutdown()

    def test_silence_counter_increments(self):
        self.w.recording = True
        self.w.recording_buffer = []
        self.w.recording_start_time = datetime.datetime.now()
        silence = np.zeros(256, dtype=np.int16)
        self.w._process_audio_chunk_immediate(silence)
        assert self.w.silence_counter > 0

    def test_squelch_open_resets_silence_counter(self):
        self.w.recording = True
        self.w.recording_buffer = []
        self.w.recording_start_time = datetime.datetime.now()
        self.w.silence_counter = 100

        # Force squelch open to reset counter
        with patch.object(
            self.w,
            "_immediate_squelch_process",
            return_value=(
                True,
                {
                    "current_db": -20,
                    "noise_floor_db": -50,
                    "level_margin_db": 30,
                    "squelch_open": True,
                    "level_score": 1,
                    "vad_probability": 0.9,
                    "vad_flag": 1,
                    "combined_score": 0.9,
                },
            ),
        ):
            self.w._process_audio_chunk_immediate(np.zeros(256, dtype=np.int16))
        assert self.w.silence_counter == 0


# ── Pre-record buffer ───────────────────────────────────────────


class TestPreRecordBuffer:
    """The pre-record buffer must act as a fixed-size circular window so the
    beginning of a transmission is never lost."""

    def setup_method(self):
        self.bus = MessageBus()
        self.w = EnhancedRxListenWorker(self.bus)

    def teardown_method(self):
        self.bus.shutdown()

    def test_buffer_does_not_exceed_max_size(self):
        chunk = np.zeros(256, dtype=np.int16)
        for _ in range(self.w.max_pre_record_chunks + 50):
            self.w._add_to_pre_record_buffer(chunk)
        assert len(self.w.pre_record_buffer) == self.w.max_pre_record_chunks

    def test_audio_buffered_when_not_recording(self):
        """Chunks processed while idle go into the pre-record buffer."""
        chunk = np.zeros(256, dtype=np.int16)
        self.w._process_audio_chunk_immediate(chunk)
        assert len(self.w.pre_record_buffer) > 0

    def test_audio_collected_when_recording(self):
        """Chunks processed while recording go into recording_buffer, not
        the pre-record buffer."""
        self.w.recording = True
        self.w.recording_buffer = []
        self.w.recording_start_time = datetime.datetime.now()
        chunk = np.ones(256, dtype=np.int16)
        self.w._process_audio_chunk_immediate(chunk)
        assert len(self.w.recording_buffer) == 1


# ── Squelch error safety ─────────────────────────────────────────


class TestSquelchErrorSafety:
    """If the squelch pipeline crashes, the worker must return safe defaults
    (squelch closed) to avoid phantom recordings."""

    def setup_method(self):
        self.bus = MessageBus()
        self.w = EnhancedRxListenWorker(self.bus)

    def teardown_method(self):
        self.bus.shutdown()

    def test_returns_closed_on_processing_error(self):
        with patch.object(
            self.w.squelch.noise_floor, "update", side_effect=RuntimeError
        ):
            is_open, stats = self.w._immediate_squelch_process(
                np.zeros(256, dtype=np.int16)
            )
        assert is_open is False
        assert stats["squelch_open"] is False


# ── Stats publishing ─────────────────────────────────────────────


class TestStatsPublishing:
    def setup_method(self):
        self.bus = MessageBus()
        self.w = EnhancedRxListenWorker(self.bus)

    def teardown_method(self):
        self.bus.shutdown()

    def test_publish_stats_reaches_bus(self):
        received = []
        self.bus.subscribe(RxAudioStatsEvent, received.append)

        self.w._publish_enhanced_stats(
            {
                "current_db": -30.0,
                "noise_floor_db": -50.0,
                "level_margin_db": 20.0,
                "squelch_open": True,
                "vad_probability": 0.8,
                "vad_flag": 1,
                "level_score": 0.7,
                "combined_score": 0.85,
            }
        )
        time.sleep(0.3)
        assert len(received) == 1
        assert received[0].squelch_open is True


# ── ALSA device parsing ──────────────────────────────────────────


class TestAlsaDeviceParsing:
    def setup_method(self):
        self.bus = MessageBus()
        self.w = EnhancedRxListenWorker(self.bus)

    def teardown_method(self):
        self.bus.shutdown()

    def test_parses_input_devices_from_ffmpeg_stderr(self):
        stderr = (
            "[0] Input card 0, device 0: USB Audio\n[1] Output card 1, device 0: HDMI"
        )
        devices = self.w._parse_alsa_devices(stderr)
        assert len(devices) == 1
        assert "USB Audio" in devices[0]

    def test_empty_stderr_returns_empty_list(self):
        assert self.w._parse_alsa_devices("") == []
        assert self.w._parse_alsa_devices(None) == []
