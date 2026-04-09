"""Tests for radiotelegram.tx — radio transmission pipeline.

Verifies:
  - TX/RX mutual exclusion: TX disabled during RX recording
  - Playback lifecycle: publishes start/end events, re-queues when disabled
  - Preprocessing failure aborts the playback pipeline
  - Full happy-path: volume → preprocess → wake tone → play → cleanup
  - Error resilience: amixer missing, ffplay timeout
"""

import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from radiotelegram.bus import MessageBus
from radiotelegram.events import (
    RxRecordingEndedEvent,
    RxRecordingStartedEvent,
    TelegramVoiceMessageDownloadedEvent,
    TxMessagePlaybackEndedEvent,
    TxMessagePlaybackStartedEvent,
)


@pytest.fixture
def bus():
    b = MessageBus()
    yield b
    b.shutdown()


@pytest.fixture
def worker(bus):
    with patch(
        "radiotelegram.tx.subprocess.run",
        return_value=MagicMock(returncode=0, stdout=""),
    ):
        from radiotelegram.tx import EnhancedTxPlayWorker

        w = EnhancedTxPlayWorker(bus)
    return w


# ── TX/RX mutual exclusion ───────────────────────────────────────


class TestTxRxMutualExclusion:
    def test_rx_recording_disables_tx(self, worker):
        worker.on_recording_started(RxRecordingStartedEvent())
        assert worker.enabled is False

    def test_rx_recording_end_enables_tx(self, worker):
        worker.enabled = False
        worker.on_recording_finished(RxRecordingEndedEvent())
        assert worker.enabled is True


# ── Playback lifecycle ───────────────────────────────────────────


class TestPlaybackLifecycle:
    @patch("radiotelegram.tx.time.sleep")
    def test_handle_event_brackets_playback_with_events(self, mock_sleep, bus, worker):
        started, ended = [], []
        bus.subscribe(TxMessagePlaybackStartedEvent, started.append)
        bus.subscribe(TxMessagePlaybackEndedEvent, ended.append)

        with patch.object(worker, "play_enhanced_audio"):
            worker.handle_event(
                TelegramVoiceMessageDownloadedEvent(filepath="/tmp/v.ogg")
            )

        time.sleep(0.3)
        assert len(started) == 1
        assert len(ended) == 1

    @patch("radiotelegram.tx.time.sleep")
    def test_disabled_worker_requeues_event(self, mock_sleep, worker):
        worker.enabled = False
        event = TelegramVoiceMessageDownloadedEvent(filepath="/tmp/v.ogg")
        with patch.object(worker, "queue_event") as mock_q:
            worker.handle_event(event)
        mock_q.assert_called_once_with(event)

    @patch("radiotelegram.tx.time.sleep")
    def test_playback_error_still_publishes_end_event(self, mock_sleep, bus, worker):
        ended = []
        bus.subscribe(TxMessagePlaybackEndedEvent, ended.append)

        with patch.object(
            worker, "play_enhanced_audio", side_effect=RuntimeError("boom")
        ):
            worker.handle_event(
                TelegramVoiceMessageDownloadedEvent(filepath="/tmp/v.ogg")
            )

        time.sleep(0.3)
        assert len(ended) == 1  # end event guarantees RX can re-enable


# ── Preprocessing pipeline ───────────────────────────────────────


class TestPreprocessing:
    def test_ffmpeg_success_returns_processed_path(self, worker):
        with patch(
            "radiotelegram.tx.subprocess.run", return_value=MagicMock(returncode=0)
        ):
            result = worker._preprocess_for_radio("/tmp/voice.ogg")
        assert result is not None and "_radio_processed" in result

    def test_ffmpeg_failure_returns_none(self, worker):
        with patch(
            "radiotelegram.tx.subprocess.run",
            return_value=MagicMock(returncode=1, stderr="codec error"),
        ):
            assert worker._preprocess_for_radio("/tmp/voice.ogg") is None

    def test_exception_returns_none(self, worker):
        with patch("radiotelegram.tx.subprocess.run", side_effect=OSError("disk full")):
            assert worker._preprocess_for_radio("/tmp/voice.ogg") is None


class TestPlayEnhancedAudioFlow:
    @patch("radiotelegram.tx.os.path.exists", return_value=True)
    @patch("radiotelegram.tx.os.remove")
    @patch("radiotelegram.tx.time.sleep")
    def test_preprocess_failure_skips_wake_tone_and_playback(
        self, mock_sleep, mock_rm, mock_exists, worker
    ):
        with (
            patch.object(worker, "_set_max_volume"),
            patch.object(worker, "_preprocess_for_radio", return_value=None),
            patch.object(worker, "_play_wake_tone") as tone,
            patch.object(worker, "_play_audio_file") as play,
        ):
            worker.play_enhanced_audio("/tmp/v.ogg")
        tone.assert_not_called()
        play.assert_not_called()

    @patch("radiotelegram.tx.os.path.exists", return_value=True)
    @patch("radiotelegram.tx.os.remove")
    @patch("radiotelegram.tx.time.sleep")
    def test_happy_path_runs_all_stages(self, mock_sleep, mock_rm, mock_exists, worker):
        with (
            patch.object(worker, "_set_max_volume") as vol,
            patch.object(
                worker, "_preprocess_for_radio", return_value="/tmp/p.ogg"
            ) as pre,
            patch.object(worker, "_play_wake_tone") as tone,
            patch.object(worker, "_play_audio_file") as play,
        ):
            worker.play_enhanced_audio("/tmp/v.ogg")
        vol.assert_called_once()
        pre.assert_called_once()
        tone.assert_called_once()
        play.assert_called_once()


# ── Error resilience ─────────────────────────────────────────────


class TestErrorResilience:
    def test_amixer_missing_does_not_crash(self, worker):
        with patch("radiotelegram.tx.subprocess.run", side_effect=FileNotFoundError):
            worker._set_max_volume()  # must not raise

    def test_ffplay_timeout_terminates_process(self, worker):
        proc = MagicMock()
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="ffplay", timeout=30)
        with patch("radiotelegram.tx.subprocess.Popen", return_value=proc):
            worker._play_audio_file("/tmp/test.ogg")
        proc.terminate.assert_called_once()
