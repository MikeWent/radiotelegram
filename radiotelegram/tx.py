import os
import subprocess
import threading
import time
from typing import Optional

from radiotelegram.bus import MessageBus, Worker, cpu_intensive
from radiotelegram.events import (
    RxRecordingEndedEvent,
    RxRecordingStartedEvent,
    TelegramVoiceMessageDownloadedEvent,
    TxMessagePlaybackEndedEvent,
    TxMessagePlaybackStartedEvent,
)


class EnhancedTxPlayWorker(Worker):
    """Enhanced transmitter worker with advanced audio processing."""

    def __init__(self, bus: MessageBus):
        super().__init__(bus)

        # Event subscriptions
        self.bus.subscribe(TelegramVoiceMessageDownloadedEvent, self.queue_event)
        self.bus.subscribe(RxRecordingStartedEvent, self.on_recording_started)
        self.bus.subscribe(RxRecordingEndedEvent, self.on_recording_finished)

        # Playback configuration
        self.enabled = True
        self.timeout_seconds = 30
        self.wake_tone_frequency = 1750  # Hz - better for radio transmission
        self.wake_tone_duration = 0.3  # seconds
        self.post_tx_delay = 2.5  # seconds for radio to return to RX mode

        # Audio processing parameters
        self.target_lufs = -16  # Loudness target
        self.limiter_ceiling = -1  # dB - prevent clipping
        self.compressor_ratio = 3.0  # Compression ratio
        self.noise_reduction_amount = 8  # dB of noise reduction

    def on_recording_started(self, event: RxRecordingStartedEvent):
        """Disable playback when recording starts to prevent interference."""
        self.enabled = False
        self.logger.debug("TX disabled - RX recording started")

    def on_recording_finished(self, event: RxRecordingEndedEvent):
        """Enable playback when recording ends."""
        self.enabled = True
        self.logger.debug("TX enabled - RX recording ended")

    def handle_event(self, event: TelegramVoiceMessageDownloadedEvent):
        """Handle incoming voice message for transmission."""
        if not self.enabled:
            self.logger.info(f"TX disabled, queuing message: {event.filepath}")
            # Re-queue the event to try later
            time.sleep(1.0)
            self.queue_event(event)
            return

        self.logger.info(f"Processing voice message for TX: {event.filepath}")
        self.bus.publish(TxMessagePlaybackStartedEvent())

        try:
            self.play_enhanced_audio(event.filepath)
        except Exception as e:
            self.logger.error(f"Error during enhanced playback: {e}")
        finally:
            time.sleep(self.post_tx_delay)
            self.bus.publish(TxMessagePlaybackEndedEvent())

    def play_enhanced_audio(self, filepath: str):
        """Play audio with enhanced processing optimized for radio transmission."""

        # Step 1: Pre-process the audio for radio transmission
        processed_filepath = self._preprocess_for_radio(filepath)

        if not processed_filepath:
            self.logger.error("Failed to preprocess audio")
            return

        try:
            # Step 2: Generate wake tone to activate radio TX
            self._play_wake_tone()

            # Step 3: Small delay for radio to fully key up
            time.sleep(0.2)

            # Step 4: Play the processed audio
            self._play_audio_file(processed_filepath)

        finally:
            # Clean up processed file
            if os.path.exists(processed_filepath):
                os.remove(processed_filepath)
            # Clean up original file
            if os.path.exists(filepath):
                os.remove(filepath)

    @cpu_intensive
    def _preprocess_for_radio(self, input_filepath: str) -> Optional[str]:
        """
        Apply comprehensive audio processing optimized for radio transmission.

        Returns path to processed file, or None if processing failed.
        """
        try:
            processed_filepath = input_filepath.replace(".ogg", "_radio_processed.ogg")

            # Comprehensive filter chain for radio transmission
            filter_complex = (
                "[0:a]"
                # 1. Normalize input level
                "loudnorm=I=-23:LRA=7:TP=-2:offset=0,"
                # 2. High-pass filter to remove rumble and handling noise
                "highpass=f=200:poles=2,"
                # 3. Pre-emphasis for radio transmission (boost highs)
                "treble=g=3:f=1000:width_type=h:width=1000,"
                # 4. Bandpass filter optimized for radio voice
                "lowpass=f=3000:poles=2,"
                # 5. Dynamic range compression for consistent levels
                "compand=attacks=0.003:decays=0.1:points=-80/-80|-43/-43|-30/-25|-18/-15:soft-knee=6:gain=0:volume=-90,"
                # 6. Gentle noise reduction
                f"afftdn=nr={self.noise_reduction_amount}:nf=-40:tn=1,"
                # 7. Peak limiter to prevent overmodulation
                f"alimiter=level_in=1:level_out=1:limit={self.limiter_ceiling}dB:attack=3:release=20,"
                # 8. Final loudness normalization for consistent TX levels
                f"loudnorm=I={self.target_lufs}:LRA=5:TP=-1[a]"
            )

            command = [
                "ffmpeg",
                "-y",
                "-i",
                input_filepath,
                "-filter_complex",
                filter_complex,
                "-map",
                "[a]",
                "-ar",
                "48000",
                "-c:a",
                "libopus",
                "-b:a",
                "96k",  # Higher quality for TX
                processed_filepath,
            ]

            self.logger.debug(f"Preprocessing audio with command: {' '.join(command)}")

            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                self.logger.debug("Audio preprocessing completed successfully")
                return processed_filepath
            else:
                self.logger.error(f"FFmpeg preprocessing failed: {result.stderr}")
                return None

        except Exception as e:
            self.logger.error(f"Error in audio preprocessing: {e}")
            return None

    def _play_wake_tone(self):
        """Play a wake-up tone to activate radio PTT reliably."""
        try:
            # Generate and play wake tone
            process = subprocess.Popen(
                [
                    "ffplay",
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    "-f",
                    "lavfi",
                    "-i",
                    f"sine=frequency={self.wake_tone_frequency}:duration={self.wake_tone_duration}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            try:
                process.wait(timeout=self.wake_tone_duration + 1.0)
                self.logger.debug(
                    f"Wake tone played: {self.wake_tone_frequency}Hz for {self.wake_tone_duration}s"
                )
            except subprocess.TimeoutExpired:
                self.logger.warning("Wake tone playback timeout")
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

        except Exception as e:
            self.logger.error(f"Error playing wake tone: {e}")

    def _play_audio_file(self, filepath: str):
        """Play processed audio file with timeout protection."""
        try:
            process = subprocess.Popen(
                [
                    "ffplay",
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    "-af",
                    "volume=0.8",  # Slight volume reduction for safety
                    filepath,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            try:
                process.wait(timeout=self.timeout_seconds)
                self.logger.debug(f"Audio playback completed: {filepath}")
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Playback timeout for {filepath}, terminating")
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.logger.warning("Force killing playback process")
                    process.kill()
                    process.wait()

        except Exception as e:
            self.logger.error(f"Error playing audio file {filepath}: {e}")

    def start(self):
        """Start processing the queue."""
        # Start the queue processing thread
        self._processing_thread = threading.Thread(
            target=self.process_queue, name=f"{self.__class__.__name__}ProcessingThread"
        )
        self._processing_thread.daemon = True
        self._processing_thread.start()
