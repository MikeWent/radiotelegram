"""Statistics worker for RX audio monitoring."""

import time

from radiotelegram.bus import MessageBus, Worker
from radiotelegram.events import RxAudioStatsEvent


class RXListenPrintStatsWorker(Worker):
    def __init__(self, bus: MessageBus):
        super().__init__(bus)
        # Subscribe to both old and new events for compatibility
        self.bus.subscribe(RxAudioStatsEvent, self.handle_audio_stats)

    def handle_audio_stats(self, event: RxAudioStatsEvent):
        # Print comprehensive audio statistics including squelch status
        performance_info = ""
        if event.avg_process_time_ms > 0 or event.max_process_time_ms > 0:
            performance_info = f", Processing: {event.avg_process_time_ms:.2f}ms avg, {event.max_process_time_ms:.2f}ms max"

        self.logger.info(
            f"Audio Stats - "
            f"Level: {event.current_db:.1f}dB, "
            f"Noise: {event.noise_floor_db:.1f}dB, "
            f"Margin: {event.level_margin_db:.1f}dB, "
            f"Squelch: {'OPEN' if event.squelch_open else 'CLOSED'}, "
            f"Voice ratio: {event.voice_ratio_score:.2f}, "
            f"Voice quality: {event.voice_quality:.2f}, "
            f"Spectral centroid: {event.spectral_centroid:.0f}Hz"
            f"{performance_info}"
        )

    def start(self):
        while True:
            time.sleep(1)
