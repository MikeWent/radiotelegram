"""Statistics worker for RX audio monitoring."""

from radiotelegram.bus import MessageBus, Worker
from radiotelegram.events import RxAudioStatsEvent


class RXListenPrintStatsWorker(Worker):
    def __init__(self, bus):
        super().__init__(bus)
        self.bus.subscribe(RxAudioStatsEvent, self.handle_audio_stats)

    def handle_audio_stats(self, event):
        perf = ""
        if event.avg_process_time_ms > 0 or event.max_process_time_ms > 0:
            perf = (
                f", Processing: {event.avg_process_time_ms:.2f}ms avg, "
                f"{event.max_process_time_ms:.2f}ms max"
            )
        self.logger.info(
            f"Level: {event.current_db:.1f}dB, "
            f"Noise: {event.noise_floor_db:.1f}dB, "
            f"Margin: {event.level_margin_db:.1f}dB, "
            f"Squelch: {'OPEN' if event.squelch_open else 'CLOSED'}, "
            f"VAD: {event.vad_probability:.2f} "
            f"({'VOICE' if event.vad_flag else 'silence'}), "
            f"Score: {event.combined_score:.2f} "
            f"(level={event.level_score:.2f} vad={event.vad_probability:.2f})"
            f"{perf}"
        )

    def start(self):
        self._stop_event.wait()
