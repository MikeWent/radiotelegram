"""Tests for radiotelegram.tx — radio transmission pipeline."""

import shutil
import struct
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
from radiotelegram.tx import EnhancedTxPlayWorker


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
        w = EnhancedTxPlayWorker(bus)
    return w


class TestWakeToneConfig:
    @pytest.fixture(autouse=True)
    def clear_wake_tone_env(self, monkeypatch):
        for name in (
            "TX_WAKE_TONE_FREQUENCY",
            "TX_WAKE_TONE_DURATION",
            "TX_WAKE_TONE_DELAY",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_defaults_allow_radio_to_key_before_speech(self, worker):
        assert worker.wake_tone_frequency == 1750
        assert worker.wake_tone_duration == 2.0
        assert worker.wake_tone_delay == 0

    def test_env_controls_generated_tone_and_playback_delay(self, monkeypatch, bus):
        monkeypatch.setenv("TX_WAKE_TONE_FREQUENCY", "1100")
        monkeypatch.setenv("TX_WAKE_TONE_DURATION", "0.8")
        monkeypatch.setenv("TX_WAKE_TONE_DELAY", "0.2")
        worker = EnhancedTxPlayWorker(bus)
        ffmpeg, aplay = MagicMock(), MagicMock()
        aplay.stderr.read.return_value = b""
        with patch("radiotelegram.tx.subprocess.Popen", side_effect=[ffmpeg, aplay]) as popen:
            worker._play_audio_file("/tmp/v.ogg")
        command = popen.call_args_list[0].args[0]
        graph = command[command.index("-filter_complex") + 1]
        assert "sine=frequency=1100.0:sample_rate=48000:duration=0.8" in graph
        assert "apad=pad_dur=0.2" in graph
        assert "concat=n=2:v=0:a=1" in graph
        assert popen.call_count == 2  # One decoder and one output for tone + voice.
        assert popen.call_args_list[1].kwargs["stdin"] is ffmpeg.stdout
        aplay.wait.assert_called_once_with(timeout=worker.timeout_seconds + 1.0)

    def test_zero_duration_skips_tone_and_delay(self, monkeypatch, bus):
        monkeypatch.setenv("TX_WAKE_TONE_DURATION", "0")
        monkeypatch.setenv("TX_WAKE_TONE_DELAY", "5")
        worker = EnhancedTxPlayWorker(bus)
        ffmpeg, aplay = MagicMock(), MagicMock()
        aplay.stderr.read.return_value = b""
        with patch("radiotelegram.tx.subprocess.Popen", side_effect=[ffmpeg, aplay]) as popen:
            worker._play_audio_file("/tmp/v.ogg")
        command = popen.call_args_list[0].args[0]
        assert "-filter_complex" not in command
        assert command[command.index("-af") + 1] == "volume=0.8"
        aplay.wait.assert_called_once_with(timeout=worker.timeout_seconds)

    @pytest.mark.parametrize(
        "name,value",
        [
            ("TX_WAKE_TONE_FREQUENCY", "invalid"),
            ("TX_WAKE_TONE_FREQUENCY", "0"),
            ("TX_WAKE_TONE_FREQUENCY", "24001"),
            ("TX_WAKE_TONE_DURATION", "-0.1"),
            ("TX_WAKE_TONE_DURATION", "inf"),
            ("TX_WAKE_TONE_DELAY", "-1"),
            ("TX_WAKE_TONE_DELAY", "nan"),
        ],
    )
    def test_invalid_env_fails_at_startup(self, name, value, monkeypatch, bus):
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError, match=name):
            EnhancedTxPlayWorker(bus)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
@pytest.mark.parametrize("delay", [0.0, 0.2])
def test_wake_stream_preserves_entire_voice(worker, tmp_path, delay):
    voice_path = tmp_path / "voice.ogg"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.25",
            "-c:a", "libopus", str(voice_path),
        ],
        check=True, capture_output=True, timeout=10,
    )

    def decode(duration):
        worker.wake_tone_duration = duration
        worker.wake_tone_delay = delay
        ffmpeg, aplay = MagicMock(), MagicMock()
        aplay.stderr.read.return_value = b""
        with patch("radiotelegram.tx.subprocess.Popen", side_effect=[ffmpeg, aplay]) as popen:
            worker._play_audio_file(str(voice_path))
        assert popen.call_count == 2
        return subprocess.run(
            popen.call_args_list[0].args[0],
            check=True, capture_output=True, timeout=10,
        ).stdout

    voice = decode(0)
    output = decode(0.8)
    tone_bytes = round(0.8 * 48000) * 2
    delay_bytes = round(delay * 48000) * 2
    assert len(voice) == 12000 * 2
    assert len(output) == tone_bytes + delay_bytes + len(voice)
    assert any(output[:tone_bytes])
    assert output[tone_bytes:tone_bytes + delay_bytes] == b"\0" * delay_bytes
    expected = struct.unpack("<12000h", voice)
    actual = struct.unpack("<12000h", output[tone_bytes + delay_bytes:])
    # Concatenation can round PCM conversion differently by one sample level.
    assert max(abs(a - b) for a, b in zip(actual, expected)) <= 1


class TestTxRxMutualExclusion:
    def test_rx_recording_disables_tx(self, worker):
        worker.on_recording_started(RxRecordingStartedEvent())
        assert worker.enabled is False

    def test_rx_recording_end_enables_tx(self, worker):
        worker.enabled = False
        worker.on_recording_finished(RxRecordingEndedEvent())
        assert worker.enabled is True


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
        assert len(ended) == 1


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
    def test_preprocess_failure_skips_playback(self, mock_sleep, mock_rm, mock_exists, worker):
        with (
            patch.object(worker, "_set_max_volume"),
            patch.object(worker, "_preprocess_for_radio", return_value=None),
            patch.object(worker, "_play_audio_file") as play,
        ):
            worker.play_enhanced_audio("/tmp/v.ogg")
        play.assert_not_called()

    @patch("radiotelegram.tx.os.path.exists", return_value=True)
    @patch("radiotelegram.tx.os.remove")
    @patch("radiotelegram.tx.time.sleep")
    def test_happy_path_runs_all_stages(self, mock_sleep, mock_rm, mock_exists, worker):
        with (
            patch.object(worker, "_set_max_volume") as vol,
            patch.object(worker, "_preprocess_for_radio", return_value="/tmp/p.ogg") as pre,
            patch.object(worker, "_play_audio_file") as play,
        ):
            worker.play_enhanced_audio("/tmp/v.ogg")
        vol.assert_called_once()
        pre.assert_called_once()
        play.assert_called_once_with("/tmp/p.ogg")
        mock_sleep.assert_not_called()


class TestErrorResilience:
    def test_amixer_missing_does_not_crash(self, worker):
        with patch("radiotelegram.tx.subprocess.run", side_effect=FileNotFoundError):
            worker._set_max_volume()

    def test_ffplay_timeout_terminates_process(self, worker):
        ffmpeg = MagicMock()
        aplay = MagicMock()
        aplay.wait.side_effect = subprocess.TimeoutExpired(cmd="aplay", timeout=30)
        aplay.stderr.read.return_value = b""
        with patch("radiotelegram.tx.subprocess.Popen", side_effect=[ffmpeg, aplay]):
            worker._play_audio_file("/tmp/test.ogg")
        aplay.terminate.assert_called_once()
        ffmpeg.terminate.assert_called_once()
