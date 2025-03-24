import asyncio
import datetime
import os
import subprocess
import time

import numpy as np
import sounddevice as sd
from bus import MessageBus, Worker
from events import (
    RxNoiseFloorStatsEvent,
    RxRecordingCompleteEvent,
    RxRecordingEndedEvent,
    RxRecordingStartedEvent,
    TelegramVoiceMessageDownloadedEvent,
    TxMessagePlaybackEndedEvent,
    TxMessagePlaybackStartedEvent,
)


class TxPlayWorker(Worker):
    def __init__(self, bus: MessageBus):
        super().__init__(bus)
        self.bus.subscribe(TelegramVoiceMessageDownloadedEvent, self.queue_event)
        self.bus.subscribe(RxRecordingStartedEvent, self.on_recording_started)
        self.bus.subscribe(RxRecordingEndedEvent, self.on_recording_finished)

        self.enabled = True
        self.timeout_seconds = 30  # Playback timeout

    async def on_recording_started(self, event: RxRecordingStartedEvent):
        """Disable playback when a new recording starts."""
        self.enabled = False

    async def on_recording_finished(self, event: RxRecordingEndedEvent):
        """Enable playback when recording stops."""
        self.enabled = True

    async def handle_event(self, event: TelegramVoiceMessageDownloadedEvent):
        self.logger.info(f"Playing voice message: {event.filepath}")
        await self.bus.publish(TxMessagePlaybackStartedEvent())
        await self.play_audio(event.filepath)
        await asyncio.sleep(2.1)  # baofeng needs time to go back to idle mode
        await self.bus.publish(TxMessagePlaybackEndedEvent())

    async def play_audio(self, filepath):
        """Play an OGG file using ffplay with a timeout."""
        # loud-normalize voice message so radio won't turn off during transmission
        normalized_filepath = filepath.replace(".ogg", "_normalized.ogg")
        command = [
            "ffmpeg",
            "-y",
            "-i",
            filepath,
            "-af",
            "loudnorm=I=-14:LRA=10:TP=-1",
            normalized_filepath,
        ]
        self.process = subprocess.Popen(command, stdout=subprocess.DEVNULL)

        try:
            with open("/dev/null", "w") as devnull:
                # play a 5555 Hz sine wave burst for 500ms to wake up baofeng
                beep_process = await asyncio.create_subprocess_exec(
                    "ffplay",
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=5555:duration=1.5",
                    stdout=devnull,
                    stderr=devnull,
                )
                await beep_process.wait()
                await asyncio.sleep(0.25)

                # play the actual audio file
                process = await asyncio.create_subprocess_exec(
                    "ffplay",
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    normalized_filepath,
                    stdout=devnull,
                )
                try:
                    await asyncio.wait_for(process.wait(), timeout=self.timeout_seconds)
                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"Playback timeout reached for {normalized_filepath}, terminating ffplay."
                    )
                    process.terminate()
                    try:
                        if process.returncode == None:
                            process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()
            os.remove(filepath)
            os.remove(normalized_filepath)
        except Exception as e:
            self.logger.error(f"Error playing audio file {filepath}: {e}")

    async def start(self):
        """Start processing the queue."""
        asyncio.create_task(self.process_queue())


class RxListenWorker(Worker):
    def __init__(self, bus: MessageBus, sample_rate=48000, chunk_size=512):
        super().__init__(bus)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_duration = 2  # Seconds to wait before stopping recording.
        self.minimal_recording_duration = (
            1  # Recordings shorter than this (after trimming) are discarded.
        )
        self.enabled = True
        self.recording = False
        self.process = None
        self.silence_counter = 0

        # Trigger: record if current dB exceeds ambient level by this many dB.
        self.noise_floor_trigger_db = 10
        # Threshold: signal should be above this absolute level
        self.noise_floor_threshold_db = -60

        # Sliding window for ambient noise in decibels (dBFS).
        self.window_size = 100
        # Pre-fill with a typical ambient level (e.g. -60 dBFS).
        self.volume_window = [self.noise_floor_threshold_db] * self.window_size

        # Publish noise floor stats every stats_interval seconds.
        self.stats_interval = 10
        self.last_stats_publish_time = 0

        self.recording_filepath = None
        self.recording_start_time = None

        self.bus.subscribe(TxMessagePlaybackStartedEvent, self.on_playback_started)
        self.bus.subscribe(TxMessagePlaybackEndedEvent, self.on_playback_finished)

    async def on_playback_started(self, event: TxMessagePlaybackStartedEvent):
        self.enabled = False

    async def on_playback_finished(self, event: TxMessagePlaybackEndedEvent):
        self.enabled = True

    def compute_db(self, data):
        """
        Compute the dBFS value for the given audio chunk.
        Uses RMS relative to full-scale (32767 for int16).
        """
        rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))
        # Add a small epsilon to avoid log(0)
        db = 20 * np.log10(rms / 32767.0 + 1e-6)
        return db

    def start_recording(self):
        if self.recording:
            return

        os.makedirs("recordings", exist_ok=True)
        filename = f"recording_{datetime.datetime.now().timestamp()}.ogg"
        self.recording_filepath = os.path.join("recordings", filename)
        self.recording_start_time = datetime.datetime.now()

        command = [
            "ffmpeg",
            "-y",
            "-f",
            "alsa",
            "-i",
            "default",
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-c:a",
            "libopus",
            self.recording_filepath,
        ]
        self.process = subprocess.Popen(
            command, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
        )
        self.recording = True
        asyncio.run(self.bus.publish(RxRecordingStartedEvent()))
        self.logger.info(f"Recording started: {self.recording_filepath}")
        self.silence_counter = 0

    def stop_recording(self):
        if not self.recording:
            return

        recording_duration = (
            datetime.datetime.now() - self.recording_start_time
        ).total_seconds()
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None

        self.recording = False
        asyncio.run(self.bus.publish(RxRecordingEndedEvent()))

        # Trim off trailing silence. Limit loudness, apply bandpass filter.
        trimmed_duration = max(0, recording_duration - self.silence_duration)
        if trimmed_duration > self.minimal_recording_duration:
            processed_filepath = self.recording_filepath.replace(
                ".ogg", "_processed.ogg"
            )
            command = [
                "ffmpeg",
                "-y",
                "-i",
                self.recording_filepath,
                "-t",
                str(trimmed_duration),
                "-ar",
                str(self.sample_rate),
                "-filter_complex",
                "[0:a]alimiter=level_in=1:level_out=1:limit=-3dB:attack=10:release=100:level=disabled, highpass=f=300, lowpass=f=8000[a]",
                "-map",
                "[a]",
                "-c:a",
                "libopus",
                processed_filepath,
            ]
            subprocess.run(command, stdout=subprocess.DEVNULL)
            asyncio.run(
                self.bus.publish(RxRecordingCompleteEvent(filepath=processed_filepath))
            )
            self.logger.info(f"Recording completed: {processed_filepath}")
        else:
            self.logger.info(
                f"Recording is too short ({trimmed_duration:.3f}s), discarded."
            )
        os.remove(self.recording_filepath)

    def process_audio_stream(self):
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_size,
        ) as stream:
            while True:
                if not self.enabled:
                    time.sleep(0.01)
                    continue

                data, _ = stream.read(self.chunk_size)

                current_db = self.compute_db(data)
                ambient_db = np.mean(self.volume_window)
                delta_db = current_db - ambient_db
                if not self.recording:
                    self.volume_window.append(current_db)
                    if len(self.volume_window) > self.window_size:
                        self.volume_window.pop(0)

                # Periodically publish noise floor statistics.
                if time.time() - self.last_stats_publish_time >= self.stats_interval:
                    asyncio.run(
                        self.bus.publish(
                            RxNoiseFloorStatsEvent(
                                ambient_db=ambient_db,
                                current_db=current_db,
                                delta_db=delta_db,
                            )
                        )
                    )
                    self.last_stats_publish_time = time.time()

                # Trigger recording if the delta exceeds the trigger value.
                if (
                    delta_db > self.noise_floor_trigger_db
                    and current_db > self.noise_floor_threshold_db
                ):
                    if not self.recording:
                        self.start_recording()
                    self.silence_counter = 0
                else:
                    if self.recording:
                        self.silence_counter += 1
                        chunks_for_silence = self.silence_duration * (
                            self.sample_rate / self.chunk_size
                        )
                        if self.silence_counter >= chunks_for_silence:
                            self.stop_recording()

    async def start(self):
        self.logger.info("RxListenWorker started")
        asyncio.create_task(asyncio.to_thread(self.process_audio_stream))


class RXListenPrintStatsWorker(Worker):
    def __init__(self, bus: MessageBus):
        super().__init__(bus)
        self.bus.subscribe(RxNoiseFloorStatsEvent, self.handle_noise_floor_stats)

    async def handle_noise_floor_stats(self, event: RxNoiseFloorStatsEvent):
        # Print or log the noise floor statistics.
        self.logger.info(
            f"Ambient: {event.ambient_db:.2f} dB, Current: {event.current_db:.2f} dB, Delta: {event.delta_db:.2f} dB"
        )

    async def start(self):
        while True:
            await asyncio.sleep(1)
