"""Tests for radiotelegram.rx_worker — RX pipeline behaviour."""

import datetime
import time
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_PATCHES = [
    patch("radiotelegram.rx_worker.subprocess.run"),
    patch("radiotelegram.rx_worker.VoiceDetector"),
    patch("radiotelegram.audio_analysis.TenVad"),
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


# ── Detection configuration ─────────────────────────────────────


class TestDetectionConfig:
    @pytest.fixture(autouse=True)
    def clear_detection_env(self, monkeypatch):
        for name in (
            "VAD_THRESHOLD",
            "VAD_MIN_VOICE_RATIO",
            "VAD_MIN_MAX_PROBABILITY",
            "VAD_MIN_AVG_PROBABILITY",
            "VAD_MIN_ANALYSIS_DURATION",
            "SQUELCH_OPEN_THRESHOLD",
            "SQUELCH_CLOSE_THRESHOLD",
            "SQUELCH_MIN_ABSOLUTE_LEVEL_DB",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_defaults_preserve_detection_settings(self, _patch_hw):
        worker = EnhancedRxListenWorker(MagicMock())
        assert worker.squelch.open_threshold == 0.65
        assert worker.squelch.close_threshold == 0.55
        assert worker.squelch.min_absolute_level == -40.0
        _patch_hw[2].assert_called_once_with(hop_size=256, threshold=0.5)
        _patch_hw[1].assert_called_once_with(
            48000,
            threshold=0.5,
            min_voice_ratio=0.01,
            min_max_probability=0.55,
            min_avg_probability=0.2,
            min_analysis_duration=0.5,
        )

    def test_env_tunes_both_detectors_and_live_hysteresis(self, monkeypatch, _patch_hw):
        for name, value in {
            "VAD_THRESHOLD": "0.4",
            "VAD_MIN_VOICE_RATIO": "0.02",
            "VAD_MIN_MAX_PROBABILITY": "0.6",
            "VAD_MIN_AVG_PROBABILITY": "0.3",
            "VAD_MIN_ANALYSIS_DURATION": "0.75",
            "SQUELCH_OPEN_THRESHOLD": "0.58",
            "SQUELCH_CLOSE_THRESHOLD": "0.48",
            "SQUELCH_MIN_ABSOLUTE_LEVEL_DB": "-45",
        }.items():
            monkeypatch.setenv(name, value)
        worker = EnhancedRxListenWorker(MagicMock())
        _patch_hw[2].assert_called_once_with(hop_size=256, threshold=0.4)
        _patch_hw[1].assert_called_once_with(
            48000,
            threshold=0.4,
            min_voice_ratio=0.02,
            min_max_probability=0.6,
            min_avg_probability=0.3,
            min_analysis_duration=0.75,
        )

        audio = np.full(768, 1000, dtype=np.int16)
        worker.squelch._vad.process.return_value = (0.2, 0)
        assert worker.squelch.process(audio)[0] is True  # score 0.6 opens
        worker.squelch._vad.process.return_value = (0.0, 0)
        assert worker.squelch.process(audio)[0] is True  # score 0.5 stays open
        is_open, stats = worker.squelch.process(np.full(768, 100, dtype=np.int16))
        assert is_open is False  # audio below configured level floor closes
        assert stats["level_score"] == 0
        assert stats["min_absolute_level"] == -45.0

    @pytest.mark.parametrize(
        "name,value",
        [
            ("VAD_THRESHOLD", "invalid"),
            ("VAD_THRESHOLD", ""),
            ("VAD_THRESHOLD", "nan"),
            ("VAD_THRESHOLD", "1.1"),
            ("VAD_MIN_VOICE_RATIO", "-0.1"),
            ("VAD_MIN_MAX_PROBABILITY", "2"),
            ("VAD_MIN_AVG_PROBABILITY", "inf"),
            ("VAD_MIN_ANALYSIS_DURATION", "-1"),
            ("VAD_MIN_ANALYSIS_DURATION", "inf"),
            ("SQUELCH_OPEN_THRESHOLD", "-0.1"),
            ("SQUELCH_CLOSE_THRESHOLD", "0.9"),
            ("SQUELCH_MIN_ABSOLUTE_LEVEL_DB", "1"),
            ("SQUELCH_MIN_ABSOLUTE_LEVEL_DB", "-inf"),
        ],
    )
    def test_invalid_env_fails_before_audio_probe(
        self, name, value, monkeypatch, _patch_hw
    ):
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError, match=name):
            EnhancedRxListenWorker(MagicMock())
        _patch_hw[0].assert_not_called()


# ── TX/RX coordination ───────────────────────────────────────────


class TestTxRxCoordination:
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

    def test_playback_end_does_not_start_duplicate_audio_thread(self):
        self.w.enabled = False
        self.w._audio_thread = MagicMock()
        self.w._audio_thread.is_alive.return_value = True
        with patch("radiotelegram.rx_worker.threading.Thread") as thread_cls:
            self.w.on_playback_finished(TxMessagePlaybackEndedEvent())
        assert self.w.enabled is True
        thread_cls.assert_not_called()

    def test_playback_during_recording_aborts_it(self):
        self.w.recording = True
        with patch.object(self.w, "stop_recording") as stop:
            self.w.on_playback_started(TxMessagePlaybackStartedEvent())
        stop.assert_called_once()
        assert self.w.enabled is False

    def test_playback_started_stops_active_stream_process(self):
        proc = MagicMock()
        proc.poll.return_value = None
        self.w.streaming_process = proc
        self.w.on_playback_started(TxMessagePlaybackStartedEvent())
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once()


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
        self.w.start_recording()
        assert self.w.recording_filepath == first_path

    def test_stale_recording_is_cleaned_up(self):
        self.w.recording = True
        self.w.recording_start_time = datetime.datetime.now() - datetime.timedelta(
            seconds=self.w.max_recording_duration + 10
        )
        self.w.recording_buffer = [np.zeros(256, dtype=np.int16)]
        with patch.object(self.w, "stop_recording", wraps=self.w.stop_recording):
            self.w.start_recording()
        assert self.w.recording is True


# ── Silence timeout ─────────────────────────────────────────────


class TestSilenceTimeout:
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
        with patch.object(
            self.w.squelch,
            "process",
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
                    "open_threshold": 0.65,
                    "close_threshold": 0.55,
                    "active_threshold": 0.55,
                    "min_absolute_level": -40.0,
                    "fast_noise_floor_db": -48.0,
                    "slow_noise_floor_db": -52.0,
                },
            ),
        ):
            self.w._process_audio_chunk_immediate(np.zeros(256, dtype=np.int16))
        assert self.w.silence_counter == 0


# ── Pre-record buffer ───────────────────────────────────────────


class TestPreRecordBuffer:
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
        chunk = np.zeros(256, dtype=np.int16)
        self.w._process_audio_chunk_immediate(chunk)
        assert len(self.w.pre_record_buffer) > 0

    def test_audio_collected_when_recording(self):
        self.w.recording = True
        self.w.recording_buffer = []
        self.w.recording_start_time = datetime.datetime.now()
        chunk = np.ones(256, dtype=np.int16)
        self.w._process_audio_chunk_immediate(chunk)
        assert len(self.w.recording_buffer) == 1


# ── Squelch error safety ─────────────────────────────────────────


class TestSquelchErrorSafety:
    def setup_method(self):
        self.bus = MessageBus()
        self.w = EnhancedRxListenWorker(self.bus)

    def teardown_method(self):
        self.bus.shutdown()

    def test_squelch_error_returns_safe_defaults(self):
        with patch.object(
            self.w.squelch.noise_floor, "update", side_effect=RuntimeError
        ):
            is_open, stats = self.w.squelch.process(np.zeros(256, dtype=np.int16))
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
        self.w._publish_stats(
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


# ── Watchdog lifecycle ─────────────────────────────────────────────


class TestWatchdogLifecycle:
    def setup_method(self):
        self.bus = MessageBus()
        self.w = EnhancedRxListenWorker(self.bus)

    def teardown_method(self):
        self.w.stop()
        self.bus.shutdown()

    def test_watchdog_survives_tx_disable_and_restarts_after_re_enable(self):
        self.w.enabled = False
        self.w._audio_thread = MagicMock()
        self.w._audio_thread.is_alive.return_value = False
        started = []
        sleep_count = 0

        def fake_sleep(seconds):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count == 2:
                self.w.enabled = True

        def fake_start(name):
            started.append(name)
            self.w._stop_event.set()
            return True

        with (
            patch("radiotelegram.rx_worker.time.sleep", side_effect=fake_sleep),
            patch.object(self.w, "_start_audio_thread", side_effect=fake_start),
        ):
            self.w._watchdog()

        assert started == ["AudioRestart"]
