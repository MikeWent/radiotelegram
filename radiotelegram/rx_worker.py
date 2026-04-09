"""RX worker: streams audio from ALSA/Pulse, applies squelch, records voice."""

import datetime
import os
import select
import subprocess
import threading
import time

import numpy as np
from scipy.io import wavfile

from radiotelegram.audio_analysis import AdvancedSquelch
from radiotelegram.bus import MessageBus, Worker
from radiotelegram.events import (
    RxAudioStatsEvent,
    RxRecordingCompleteEvent,
    RxRecordingEndedEvent,
    RxRecordingStartedEvent,
    TxMessagePlaybackEndedEvent,
    TxMessagePlaybackStartedEvent,
)
from radiotelegram.voice_detection import VoiceDetector


class EnhancedRxListenWorker(Worker):
    def __init__(self, bus, sample_rate=48000, chunk_size=256, audio_device="hw:1,0"):
        super().__init__(bus)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_device = audio_device

        # Recording
        self.silence_duration = 2.0
        self.minimal_recording_duration = 1.0
        self.max_recording_duration = 60.0
        self.recording = False
        self.silence_counter = 0
        self.recording_filepath = None
        self.recording_start_time = None
        self.recording_buffer = []

        # Pre-record buffer
        self.pre_record_duration = 0.5
        self.pre_record_buffer = []
        self.max_pre_record_chunks = int(
            self.pre_record_duration * sample_rate / chunk_size
        )
        self.pre_record_filepath = None

        # Processing
        self.squelch = AdvancedSquelch(sample_rate)
        self.voice_detector = VoiceDetector(sample_rate)

        # Stats
        self.stats_interval = 2.0
        self.last_stats_publish_time = 0
        self.chunk_process_times = []

        # Streaming
        self.streaming_process = None
        self.last_chunk_time = 0
        self._audio_thread = None

        # Events
        self.bus.subscribe(TxMessagePlaybackStartedEvent, self.on_playback_started)
        self.bus.subscribe(TxMessagePlaybackEndedEvent, self.on_playback_finished)

        self._probe_audio_device()

    # ── TX/RX coordination ─────────────────────────────────────

    def on_playback_started(self, event):
        self.enabled = False
        if self.recording:
            self.logger.info("Stopping recording due to TX playback")
            self.stop_recording()

    def on_playback_finished(self, event):
        self.enabled = True
        self.logger.info("TX finished, restarting audio stream")
        t = threading.Thread(
            target=self.process_audio_stream, daemon=True, name="AudioStreamThread"
        )
        t.start()

    # ── Device probing ─────────────────────────────────────────

    def _probe_audio_device(self):
        if self._test_device(self.audio_device):
            self.logger.info(f"Audio device '{self.audio_device}' OK")
            return
        for dev in ["hw:1,0", "plughw:1,0", "hw:0,0", "plughw:0,0", "pulse"]:
            if dev != self.audio_device and self._test_device(dev):
                self.logger.info(f"Switched to audio device: {dev}")
                self.audio_device = dev
                return
        self.logger.warning("No working audio device found")

    def _test_device(self, device):
        try:
            test_file = "/tmp/audio_test.wav"
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-f", "alsa", "-i", device,
                    "-t", "0.1", "-ar", str(self.sample_rate),
                    "-ac", "1", "-c:a", "pcm_s16le", test_file,
                ],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and os.path.exists(test_file):
                os.remove(test_file)
                return True
        except (subprocess.TimeoutExpired, Exception):
            pass
        return False

    # ── Pre-record buffer ──────────────────────────────────────

    def _add_to_pre_record_buffer(self, chunk):
        self.pre_record_buffer.append(chunk.tobytes())
        if len(self.pre_record_buffer) > self.max_pre_record_chunks:
            self.pre_record_buffer.pop(0)

    def _save_pre_record_buffer(self, filepath):
        if not self.pre_record_buffer:
            return False
        try:
            audio = np.frombuffer(b"".join(self.pre_record_buffer), dtype=np.int16)
            wavfile.write(filepath, self.sample_rate, audio)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save pre-record buffer: {e}")
            return False

    # ── Recording lifecycle ────────────────────────────────────

    def start_recording(self):
        if self.recording:
            if (
                self.recording_start_time
                and (datetime.datetime.now() - self.recording_start_time).total_seconds()
                > self.max_recording_duration
            ):
                self.logger.warning("Cleaning up stale recording")
                self.stop_recording()
            else:
                return

        timestamp = datetime.datetime.now().timestamp()

        pre_path = f"/tmp/pre_record_{timestamp:.3f}.wav"
        self.pre_record_filepath = (
            pre_path if self._save_pre_record_buffer(pre_path) else None
        )

        self.recording_filepath = f"/tmp/recording_{timestamp:.3f}.wav"
        self.recording_start_time = datetime.datetime.now()
        self.recording = True
        self.silence_counter = 0
        self.recording_buffer = []

        self.bus.publish(RxRecordingStartedEvent())
        self.logger.info(f"Recording started: {self.recording_filepath}")

    def stop_recording(self):
        if not self.recording or not self.recording_start_time:
            return

        duration = (
            datetime.datetime.now() - self.recording_start_time
        ).total_seconds()
        self.logger.info(f"Stopping recording after {duration:.1f}s")

        if self.recording_buffer:
            try:
                wavfile.write(
                    self.recording_filepath,
                    self.sample_rate,
                    np.concatenate(self.recording_buffer),
                )
            except Exception as e:
                self.logger.error(f"Failed to save recording: {e}")
                self.recording = False
                self.silence_counter = 0
                self.bus.publish(RxRecordingEndedEvent())
                return

        self.recording = False
        self.silence_counter = 0
        self.bus.publish(RxRecordingEndedEvent())

        if not self.recording_filepath or not os.path.exists(self.recording_filepath):
            return

        if os.path.getsize(self.recording_filepath) < 1024:
            self.logger.warning("Recording too small, discarding")
            os.remove(self.recording_filepath)
            return

        trimmed = max(0, duration - self.silence_duration)
        if trimmed > self.minimal_recording_duration:
            self._process_recording()
        else:
            self.logger.info(f"Recording too short ({trimmed:.1f}s), discarded")
            if os.path.exists(self.recording_filepath):
                os.remove(self.recording_filepath)

    def _process_recording(self):
        """Merge pre-record buffer, apply FFmpeg filters, run voice detection."""
        if not self.recording_filepath or not os.path.exists(self.recording_filepath):
            return

        try:
            merged = self.recording_filepath

            if self.pre_record_filepath and os.path.exists(self.pre_record_filepath):
                merged = self.recording_filepath.replace(".wav", "_merged.wav")
                result = subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", self.pre_record_filepath,
                        "-i", self.recording_filepath,
                        "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[a]",
                        "-map", "[a]", "-c:a", "pcm_s16le", merged,
                    ],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    text=True, timeout=15,
                )
                if result.returncode != 0:
                    merged = self.recording_filepath
                try:
                    os.remove(self.pre_record_filepath)
                except OSError:
                    pass

            processed = merged.replace(".wav", "_processed.ogg")
            filter_chain = (
                "[0:a]highpass=f=300,"
                "treble=g=-6:f=1000:width_type=h:width=1000,"
                "lowpass=f=3400,"
                "compand=attacks=0.01:decays=0.5:"
                "points=-70/-70|-60/-50|-30/-20|-10/-10:"
                "soft-knee=6:gain=0:volume=-20,"
                "afftdn=nr=10:nf=-50:tn=1,"
                "alimiter=level_in=1:level_out=1:limit=-1dB:"
                "attack=5:release=50[a]"
            )
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", merged,
                    "-filter_complex", filter_chain, "-map", "[a]",
                    "-ar", str(self.sample_rate),
                    "-c:a", "libopus", "-b:a", "64k", processed,
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True, timeout=30,
            )

            if result.returncode != 0:
                self.logger.error(f"FFmpeg processing failed: {result.stderr}")
                return

            is_voice, analysis = self.voice_detector.analyze_recording(processed)
            self.logger.info(
                f"Voice analysis: voice={is_voice}, "
                f"ratio={analysis.get('voice_ratio', 0):.1%}, "
                f"max_prob={analysis.get('max_probability', 0):.3f}"
            )

            if is_voice:
                self.bus.publish(RxRecordingCompleteEvent(filepath=processed))
            else:
                self.logger.info("No voice detected, discarding")
                if os.path.exists(processed):
                    os.remove(processed)

        except (subprocess.TimeoutExpired, Exception) as e:
            self.logger.error(f"Error processing recording: {e}")
        finally:
            for path in [
                self.recording_filepath,
                self.recording_filepath.replace(".wav", "_merged.wav"),
            ]:
                if os.path.exists(path):
                    os.remove(path)

    # ── Real-time audio processing ─────────────────────────────

    def _process_audio_chunk_immediate(self, audio_chunk):
        start_time = time.time()
        try:
            if not self.recording:
                self._add_to_pre_record_buffer(audio_chunk)
            elif self.recording_buffer is not None:
                self.recording_buffer.append(audio_chunk.copy())

            squelch_open, stats = self.squelch.process(audio_chunk)
            current_time = time.time()

            if squelch_open and not self.recording:
                self.start_recording()
            elif not squelch_open and self.recording:
                self.silence_counter += 1
                chunks_for_silence = self.silence_duration * (
                    self.sample_rate / self.chunk_size
                )
                if self.silence_counter >= chunks_for_silence:
                    self.logger.info("Silence timeout, stopping recording")
                    self.stop_recording()
            elif squelch_open and self.recording:
                self.silence_counter = 0
                if (
                    self.recording_start_time
                    and (datetime.datetime.now() - self.recording_start_time).total_seconds()
                    > self.max_recording_duration
                ):
                    self.logger.warning("Recording timeout, stopping")
                    self.stop_recording()

            # Performance + stats
            self.chunk_process_times.append(time.time() - start_time)
            if len(self.chunk_process_times) > 100:
                self.chunk_process_times.pop(0)

            if current_time - self.last_stats_publish_time >= self.stats_interval:
                stats["avg_process_time_ms"] = (
                    sum(self.chunk_process_times) / len(self.chunk_process_times) * 1000
                )
                stats["max_process_time_ms"] = max(self.chunk_process_times) * 1000
                self._publish_stats(stats)
                self.last_stats_publish_time = current_time

        except Exception as e:
            self.logger.error(f"Error processing audio chunk: {e}")

    def _publish_stats(self, stats):
        self.bus.publish(
            RxAudioStatsEvent(
                current_db=float(stats["current_db"]),
                noise_floor_db=float(stats["noise_floor_db"]),
                level_margin_db=float(stats["level_margin_db"]),
                squelch_open=bool(stats["squelch_open"]),
                vad_probability=float(stats.get("vad_probability", 0)),
                vad_flag=int(stats.get("vad_flag", 0)),
                level_score=float(stats.get("level_score", 0)),
                combined_score=float(stats.get("combined_score", 0)),
                avg_process_time_ms=float(stats.get("avg_process_time_ms", 0)),
                max_process_time_ms=float(stats.get("max_process_time_ms", 0)),
            )
        )

    # ── FFmpeg audio streaming ─────────────────────────────────

    def process_audio_stream(self):
        for attempt in range(5):
            if not self.enabled:
                return
            self.logger.info(
                f"Audio stream attempt {attempt + 1}/5 on {self.audio_device}"
            )
            if self._stream_audio_once():
                break
            time.sleep(min(2.0 * (2 ** attempt), 10))
        else:
            self.logger.error("All audio stream attempts failed")

    def _stream_audio_once(self):
        cmd = [
            "ffmpeg", "-f", "alsa", "-thread_queue_size", "512",
            "-i", self.audio_device, "-ac", "1", "-ar", str(self.sample_rate),
            "-acodec", "pcm_s16le", "-fflags", "nobuffer",
            "-flags", "low_delay", "-strict", "experimental",
            "-f", "s16le", "-",
        ]
        process = None
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
            )
            self.streaming_process = process
            if not process.stdout:
                return False

            chunk_bytes = min(256, self.chunk_size) * 2
            audio_buffer = bytearray()
            chunks_ok = 0

            while self.enabled:
                if process.poll() is not None:
                    _, stderr = process.communicate(timeout=1)
                    self.logger.warning(
                        f"FFmpeg ended: {stderr.decode()[:200] if stderr else ''}"
                    )
                    return chunks_ok >= 10

                ready, _, _ = select.select([process.stdout], [], [], 0.01)
                if not ready:
                    continue

                raw = process.stdout.read(chunk_bytes)
                if not raw:
                    if process.poll() is not None:
                        break
                    continue

                audio_buffer.extend(raw)
                while len(audio_buffer) >= chunk_bytes:
                    chunk = np.frombuffer(audio_buffer[:chunk_bytes], dtype=np.int16)
                    audio_buffer = audio_buffer[chunk_bytes:]
                    self._process_audio_chunk_immediate(chunk)
                    chunks_ok += 1
                    self.last_chunk_time = time.time()

            return chunks_ok >= 10
        except Exception as e:
            self.logger.error(f"Audio stream error: {e}")
            return False
        finally:
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            self.streaming_process = None

    # ── Watchdog ───────────────────────────────────────────────

    def _watchdog(self):
        while self.enabled:
            time.sleep(5.0)
            if (
                self.last_chunk_time
                and time.time() - self.last_chunk_time > 30
                and self.enabled
            ):
                self.logger.warning("Audio stream stalled, restarting")
                if self.streaming_process and self.streaming_process.poll() is None:
                    self.streaming_process.kill()
                    try:
                        self.streaming_process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                self.last_chunk_time = 0
                self._audio_thread = threading.Thread(
                    target=self.process_audio_stream,
                    daemon=True, name="AudioRestart",
                )
                self._audio_thread.start()
            elif (
                self._audio_thread
                and not self._audio_thread.is_alive()
                and self.enabled
            ):
                self.logger.warning("Audio thread died, restarting")
                self._audio_thread = threading.Thread(
                    target=self.process_audio_stream,
                    daemon=True, name="AudioRestart",
                )
                self._audio_thread.start()

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self):
        self.logger.info("RxListenWorker starting")
        self._audio_thread = threading.Thread(
            target=self.process_audio_stream, daemon=True, name="AudioStream"
        )
        self._audio_thread.start()
        threading.Thread(
            target=self._watchdog, daemon=True, name="AudioWatchdog"
        ).start()

    def stop(self):
        self.logger.info("Stopping RxListenWorker")
        self.enabled = False
        if self.recording:
            self.stop_recording()
        if self.streaming_process and self.streaming_process.poll() is None:
            self.streaming_process.terminate()
            try:
                self.streaming_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.streaming_process.kill()
        self.streaming_process = None
        super().stop()
