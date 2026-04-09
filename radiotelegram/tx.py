import os
import subprocess
import threading
import time
from typing import Optional

from radiotelegram.bus import MessageBus, Worker
from radiotelegram.events import (
    RxRecordingEndedEvent,
    RxRecordingStartedEvent,
    TelegramVoiceMessageDownloadedEvent,
    TxMessagePlaybackEndedEvent,
    TxMessagePlaybackStartedEvent,
)


class EnhancedTxPlayWorker(Worker):
    """Transmitter: processes and plays voice messages over radio."""

    def __init__(self, bus):
        super().__init__(bus)

        self.bus.subscribe(TelegramVoiceMessageDownloadedEvent, self.queue_event)
        self.bus.subscribe(RxRecordingStartedEvent, self.on_recording_started)
        self.bus.subscribe(RxRecordingEndedEvent, self.on_recording_finished)

        self.timeout_seconds = 30
        self.wake_tone_frequency = 1750
        self.wake_tone_duration = 0.3
        self.post_tx_delay = 2.5

        self.target_lufs = -16
        self.limiter_ceiling = -1
        self.noise_reduction_amount = 8
        self.max_volume_enabled = True
        self.volume_control_device = "PCM"

        self._log_output_device_info()

    def _log_output_device_info(self):
        try:
            result = subprocess.run(
                ["aplay", "-l"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("card "):
                        self.logger.info(f"TX output device: {line.strip()}")
                        break
        except Exception as e:
            self.logger.warning(f"Failed to query TX output device: {e}")

    def on_recording_started(self, event):
        self.enabled = False

    def on_recording_finished(self, event):
        self.enabled = True

    def handle_event(self, event):
        if not self.enabled:
            self.logger.info(f"TX disabled, re-queuing: {event.filepath}")
            time.sleep(1.0)
            self.queue_event(event)
            return

        self.logger.info(f"TX playing: {event.filepath}")
        self.bus.publish(TxMessagePlaybackStartedEvent())
        try:
            self.play_enhanced_audio(event.filepath)
        except Exception as e:
            self.logger.error(f"Playback error: {e}")
        finally:
            time.sleep(self.post_tx_delay)
            self.bus.publish(TxMessagePlaybackEndedEvent())

    def play_enhanced_audio(self, filepath):
        if self.max_volume_enabled:
            self._set_max_volume()

        processed = self._preprocess_for_radio(filepath)
        if not processed:
            self.logger.error("Preprocessing failed")
            return

        try:
            self._play_wake_tone()
            time.sleep(0.2)
            self._play_audio_file(processed)
        finally:
            for f in [processed, filepath]:
                if os.path.exists(f):
                    os.remove(f)

    def _preprocess_for_radio(self, input_filepath) -> Optional[str]:
        try:
            out = input_filepath.replace(".ogg", "_radio_processed.ogg")
            filter_complex = (
                "[0:a]"
                "loudnorm=I=-23:LRA=7:TP=-2:offset=0,"
                "highpass=f=200:poles=2,"
                "treble=g=3:f=1000:width_type=h:width=1000,"
                "lowpass=f=3000:poles=2,"
                "compand=attacks=0.003:decays=0.1:"
                "points=-80/-80|-43/-43|-30/-25|-18/-15:"
                "soft-knee=6:gain=0:volume=-90,"
                f"afftdn=nr={self.noise_reduction_amount}:nf=-40:tn=1,"
                f"alimiter=level_in=1:level_out=1:limit={self.limiter_ceiling}dB:"
                "attack=3:release=20,"
                f"loudnorm=I={self.target_lufs}:LRA=5:TP=-1[a]"
            )
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", input_filepath,
                    "-filter_complex", filter_complex, "-map", "[a]",
                    "-ar", "48000", "-c:a", "libopus", "-b:a", "96k", out,
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True, timeout=30,
            )
            return out if result.returncode == 0 else None
        except Exception as e:
            self.logger.error(f"Preprocessing error: {e}")
            return None

    def _set_max_volume(self):
        try:
            subprocess.run(
                ["amixer", "sset", self.volume_control_device, "100%"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

    def _play_wake_tone(self):
        try:
            proc = subprocess.Popen(
                [
                    "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                    "-f", "lavfi", "-i",
                    f"sine=frequency={self.wake_tone_frequency}:"
                    f"duration={self.wake_tone_duration}",
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            try:
                proc.wait(timeout=self.wake_tone_duration + 1.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception as e:
            self.logger.error(f"Wake tone error: {e}")

    def _play_audio_file(self, filepath):
        try:
            proc = subprocess.Popen(
                [
                    "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                    "-af", "volume=0.8", filepath,
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            try:
                proc.wait(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception as e:
            self.logger.error(f"Playback error: {e}")

    def start(self):
        t = threading.Thread(
            target=self.process_queue, daemon=True,
            name=f"{self.__class__.__name__}Queue",
        )
        t.start()
