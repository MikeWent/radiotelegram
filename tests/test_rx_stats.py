"""Tests for radiotelegram.rx_stats — stats logging behaviour.

Verifies that RXListenPrintStatsWorker formats log lines correctly depending
on squelch state, VAD flag, and whether performance metrics are present.
"""

import logging

from radiotelegram.bus import MessageBus
from radiotelegram.events import RxAudioStatsEvent
from radiotelegram.rx_stats import RXListenPrintStatsWorker


class TestStatsLogOutput:
    def setup_method(self):
        self.bus = MessageBus()
        self.worker = RXListenPrintStatsWorker(self.bus)

    def teardown_method(self):
        self.bus.shutdown()

    def test_squelch_open_voice_detected_log(self, caplog):
        event = RxAudioStatsEvent(
            current_db=-30.0,
            noise_floor_db=-50.0,
            level_margin_db=20.0,
            squelch_open=True,
            vad_probability=0.8,
            vad_flag=1,
            level_score=0.7,
            combined_score=0.85,
            open_threshold=0.65,
            close_threshold=0.55,
            active_threshold=0.55,
            min_absolute_level=-40.0,
            fast_noise_floor_db=-48.0,
            slow_noise_floor_db=-52.0,
        )
        with caplog.at_level(logging.INFO, logger="RXListenPrintStatsWorker"):
            self.worker.handle_audio_stats(event)

        assert "OPEN" in caplog.text
        assert "VOICE" in caplog.text
        assert "Level: -30.0dB" in caplog.text
        assert "thr=0.55" in caplog.text
        assert "fast=-48.0" in caplog.text

    def test_squelch_closed_silence_log(self, caplog):
        event = RxAudioStatsEvent(
            current_db=-55.0,
            noise_floor_db=-50.0,
            level_margin_db=-5.0,
            squelch_open=False,
            vad_probability=0.1,
            vad_flag=0,
        )
        with caplog.at_level(logging.INFO, logger="RXListenPrintStatsWorker"):
            self.worker.handle_audio_stats(event)

        assert "CLOSED" in caplog.text
        assert "silence" in caplog.text

    def test_performance_metrics_included_when_nonzero(self, caplog):
        event = RxAudioStatsEvent(
            current_db=-30.0,
            noise_floor_db=-50.0,
            level_margin_db=20.0,
            squelch_open=True,
            vad_probability=0.5,
            vad_flag=0,
            avg_process_time_ms=2.5,
            max_process_time_ms=5.0,
        )
        with caplog.at_level(logging.INFO, logger="RXListenPrintStatsWorker"):
            self.worker.handle_audio_stats(event)

        assert "Processing:" in caplog.text
        assert "2.50ms avg" in caplog.text

    def test_performance_metrics_omitted_when_zero(self, caplog):
        event = RxAudioStatsEvent(
            current_db=-30.0,
            noise_floor_db=-50.0,
            level_margin_db=20.0,
            squelch_open=False,
            vad_probability=0.0,
            vad_flag=0,
        )
        with caplog.at_level(logging.INFO, logger="RXListenPrintStatsWorker"):
            self.worker.handle_audio_stats(event)

        assert "Processing:" not in caplog.text

    def test_bus_delivers_stats_to_worker(self):
        """Publishing an RxAudioStatsEvent on the bus reaches the worker."""
        received = []
        original = self.worker.handle_audio_stats
        self.worker.handle_audio_stats = lambda e: received.append(e)
        # Re-subscribe with the patched method
        self.bus.subscribers[RxAudioStatsEvent] = [self.worker.handle_audio_stats]

        self.bus.publish(
            RxAudioStatsEvent(
                current_db=-40.0,
                noise_floor_db=-50.0,
                level_margin_db=10.0,
                squelch_open=False,
                vad_probability=0.0,
                vad_flag=0,
            )
        )
        assert len(received) == 1
