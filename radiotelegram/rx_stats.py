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
        rec = " [REC]" if event.is_recording else ""
        self.logger.info(
            f"Level: {event.current_db:.1f}dB, "
            f"Noise: {event.noise_floor_db:.1f}dB "
            f"(fast={event.fast_noise_floor_db:.1f} slow={event.slow_noise_floor_db:.1f}), "
            f"Margin: {event.level_margin_db:.1f}dB, "
            f"Squelch: {'OPEN' if event.squelch_open else 'CLOSED'} "
            f"(score={event.combined_score:.2f} "
            f"thr={event.active_threshold:.2f} "
            f"open={event.open_threshold:.2f} close={event.close_threshold:.2f}), "
            f"VAD: {event.vad_probability:.2f} "
            f"({'VOICE' if event.vad_flag else 'silence'}), "
            f"Level score: {event.level_score:.2f}, "
            f"Min abs: {event.min_absolute_level:.0f}dB"
            f"{rec}{perf}"
        )

    def start(self):
        self._stop_event.wait()
