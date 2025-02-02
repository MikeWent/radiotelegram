import asyncio
import datetime
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import sounddevice as sd

from bus import (
    MessageBus,
    RxRecordingCompleteEvent,
    RxRecordingEndedEvent,
    RxRecordingStartedEvent,
    TelegramVoiceMessageDownloadedEvent,
    TxMessagePlaybackEndedEvent,
    TxMessagePlaybackStartedEvent,
    Worker,
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
                    "sine=frequency=5555:duration=1",
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
                    filepath,
                    stdout=devnull,
                )
                try:
                    await asyncio.wait_for(process.wait(), timeout=self.timeout_seconds)
                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"Playback timeout reached for {filepath}, terminating ffplay."
                    )
                    process.terminate()
                    try:
                        if process.returncode == None:
                            process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()
            os.remove(filepath)
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
        self.silence_duration = 2
        self.minimal_recording_duration = 1
        self.enabled = True
        self.recording = False
        self.process = None
        self.threshold = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.recording_filepath = None
        self.recording_start_time = None

        self.bus.subscribe(TxMessagePlaybackStartedEvent, self.on_playback_started)
        self.bus.subscribe(TxMessagePlaybackEndedEvent, self.on_playback_finished)

    async def on_playback_started(self, event: TxMessagePlaybackStartedEvent):
        self.enabled = False

    async def on_playback_finished(self, event: TxMessagePlaybackEndedEvent):
        self.enabled = True

    def calibrate_threshold(self, duration=2):
        """Listen to the environment and determine a suitable silence threshold."""
        if self.threshold is not None:
            return  # Prevent recalibration if already set

        self.logger.info("Calibrating silence threshold...")
        audio = sd.rec(
            int(self.sample_rate * duration),
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        self.threshold = (
            np.mean(np.abs(audio)) * 3
        )  # Set threshold slightly above ambient noise
        self.logger.info(f"Calibrated silence threshold: {self.threshold}")

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

        trimmed_duration = max(0, recording_duration - self.silence_duration)
        if trimmed_duration > self.minimal_recording_duration:
            trimmed_filepath = self.recording_filepath.replace(".ogg", "_trimmed.ogg")
            # Remove last seconds of silence
            command = [
                "ffmpeg",
                "-y",
                "-i",
                self.recording_filepath,
                "-t",
                str(trimmed_duration),
                "-c",
                "copy",
                trimmed_filepath,
            ]
            subprocess.run(
                command, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
            )

            while not (os.path.exists(trimmed_filepath)):
                time.sleep(0.1)
            
            # Replace original file with trimmed file
            os.replace(trimmed_filepath, self.recording_filepath)

            asyncio.run(
                self.bus.publish(
                    RxRecordingCompleteEvent(filepath=self.recording_filepath)
                )
            )
            self.logger.info(f"Recording completed: {self.recording_filepath}")
        else:
            os.remove(self.recording_filepath)
            self.logger.info(
                f"Recording is too short ({trimmed_duration:.3f}s), discarded."
            )

    def process_audio_stream(self):
        """Monitor microphone activity and control ffmpeg recording based on silence detection."""
        self.calibrate_threshold()
        silence_counter = 0
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_size,
        ) as stream:
            while True:
                if not self.enabled:
                    continue

                data, _ = stream.read(self.chunk_size)
                volume = np.mean(np.abs(data))

                if volume > self.threshold:
                    if not self.recording:
                        self.start_recording()
                    silence_counter = 0
                elif self.recording:
                    silence_counter += 1
                    if silence_counter >= (
                        self.silence_duration * self.sample_rate / self.chunk_size
                    ):
                        self.stop_recording()

    async def start(self):
        self.logger.info("RxListenWorker started")
        await asyncio.to_thread(self.calibrate_threshold)
        # don't block async loop with audio processing
        asyncio.create_task(asyncio.to_thread(self.process_audio_stream))
