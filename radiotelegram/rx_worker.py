"""Enhanced RX worker for audio recording and processing."""

import datetime
import os
import select
import subprocess
import threading
import time
from typing import Optional, Tuple

import numpy as np
from scipy.io import wavfile

from radiotelegram.audio_analysis import AdvancedSquelch
from radiotelegram.bus import MessageBus, Worker, cpu_intensive
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
    """Enhanced receiver worker with advanced audio processing."""

    def __init__(
        self, bus: MessageBus, sample_rate=48000, chunk_size=256, audio_device="pulse"
    ):
        super().__init__(bus)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size  # Reduced default chunk size for lower latency
        self.audio_device = (
            audio_device  # Default to pulse (PipeWire/PulseAudio compatible)
        )

        # Recording parameters optimized for responsive squelch
        self.silence_duration = 2.0  # Seconds to wait before stopping recording
        self.minimal_recording_duration = 1.0  # Min duration after trimming
        self.max_recording_duration = 60.0  # Maximum recording duration (timeout)
        self.enabled = True
        self.recording = False
        self.process = None
        self.silence_counter = 0

        # Squelch state tracking - optimized for immediate response
        self.squelch_open_time: Optional[float] = None
        self.min_squelch_open_duration = 0.5

        # Pre-recording buffer to capture audio before squelch opens
        self.pre_record_duration = 0.5  # Seconds of audio to buffer before squelch
        self.pre_record_buffer = []  # Circular buffer for audio chunks
        self.max_pre_record_chunks = int(
            self.pre_record_duration * sample_rate / chunk_size
        )

        # File paths
        self.recording_filepath: Optional[str] = None
        self.pre_record_filepath: Optional[str] = None
        self.recording_start_time: Optional[datetime.datetime] = None
        self.recording_buffer = []  # Buffer to collect audio during recording

        # Advanced audio processing
        self.squelch = AdvancedSquelch(sample_rate)
        self.voice_detector = VoiceDetector(sample_rate)

        # Statistics and monitoring - reduced interval for better responsiveness monitoring
        self.stats_interval = 2.0  # Publish stats every 2 seconds (reduced from 5s)
        self.last_stats_publish_time = 0

        # Performance monitoring for latency debugging
        self.chunk_process_times = []
        self.max_process_time_samples = 100  # Keep last 100 samples

        # Audio streaming process monitoring
        self.streaming_process = None  # Track the streaming FFmpeg process
        self.last_chunk_time = 0  # Track when we last received audio data
        self.streaming_timeout = (
            10.0  # Seconds without audio before considering stream dead
        )

        # Store reference for thread management
        self._processing_thread = None

        # Event subscriptions
        self.bus.subscribe(TxMessagePlaybackStartedEvent, self.on_playback_started)
        self.bus.subscribe(TxMessagePlaybackEndedEvent, self.on_playback_finished)

        # Test audio device availability at startup
        self._test_audio_device()

    def on_playback_started(self, event: TxMessagePlaybackStartedEvent):
        """Disable listening during playback to prevent feedback."""
        self.enabled = False
        if self.recording:
            self.logger.info("Stopping recording due to TX playback starting")
            self.stop_recording()

    def on_playback_finished(self, event: TxMessagePlaybackEndedEvent):
        """Re-enable listening after playback ends."""
        self.enabled = True
        # Restart audio stream processing
        self.logger.info("TX playback finished, restarting audio stream processing")
        # Start audio processing in a new thread
        processing_thread = threading.Thread(
            target=self.process_audio_stream, name="AudioProcessingThread"
        )
        processing_thread.daemon = True
        processing_thread.start()

    def _test_audio_device(self):
        """Test audio device availability and log diagnostics."""
        # Test if we can list audio devices
        result = subprocess.run(
            ["ffmpeg", "-f", "alsa", "-list_devices", "true", "-i", "dummy"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.stderr:
            self.logger.debug(f"Available ALSA devices:\n{result.stderr}")

        # Extract available input devices from the output
        available_devices = self._parse_alsa_devices(result.stderr)
        self.logger.info(f"Detected ALSA input devices: {available_devices}")

        # Test the configured device first
        if self._test_device(self.audio_device):
            self.logger.info(f"Audio device '{self.audio_device}' is working")
            return

        # If default device fails, try other common device names
        fallback_devices = [
            "pulse",  # PulseAudio/PipeWire compatibility layer (try first)
            "pipewire",  # Direct PipeWire access
            "hw:0,0",
            "plughw:0,0",
            "hw:1,0",
            "plughw:1,0",
            "hw:2,0",
            "plughw:2,0",  # Try more hardware devices
            "hw:0",
            "plughw:0",  # Simplified hardware references
            "hw:1",
            "plughw:1",
        ]

        for device in fallback_devices:
            if device != self.audio_device:  # Don't test the same device twice
                self.logger.info(f"Testing fallback audio device: {device}")
                if self._test_device(device):
                    self.logger.info(f"Switching to working audio device: {device}")
                    self.audio_device = device
                    return

    def _add_to_pre_record_buffer(self, audio_chunk: np.ndarray):
        """Add audio chunk to the pre-recording circular buffer."""
        # Convert to bytes for consistent storage
        audio_bytes = audio_chunk.tobytes()

        # Add to circular buffer
        self.pre_record_buffer.append(audio_bytes)

        # Maintain buffer size
        if len(self.pre_record_buffer) > self.max_pre_record_chunks:
            self.pre_record_buffer.pop(0)

    @cpu_intensive
    def _save_pre_record_buffer(self, filepath: str):
        """Save the pre-recording buffer to a WAV file."""
        if not self.pre_record_buffer:
            self.logger.warning("No pre-record buffer to save")
            return False

        try:
            # Combine all buffered chunks
            combined_bytes = b"".join(self.pre_record_buffer)

            # Convert bytes back to numpy array
            audio_data = np.frombuffer(combined_bytes, dtype=np.int16)

            # Save as WAV file
            wavfile.write(filepath, self.sample_rate, audio_data)

            file_size = os.path.getsize(filepath)
            duration = len(audio_data) / self.sample_rate
            self.logger.debug(
                f"Saved pre-record buffer: {file_size} bytes, {duration:.3f}s"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to save pre-record buffer: {e}")
            return False

    def _parse_alsa_devices(self, stderr_output):
        """Parse FFmpeg ALSA device list output to extract available devices."""
        devices = []
        if not stderr_output:
            return devices

        lines = stderr_output.split("\n")
        for line in lines:
            # Look for input device lines like "[0] card 0, device 0: ..."
            if "Input" in line and "card" in line:
                # Extract device identifier
                if "[" in line and "]" in line:
                    device_info = line.split("]", 1)[1].strip()
                    devices.append(device_info)
        return devices

    def _test_device(self, device):
        """Test if a specific audio device works by recording a short sample."""
        try:
            test_file = "/tmp/audio_test.wav"
            test_command = [
                "ffmpeg",
                "-y",
                "-f",
                "alsa",
                "-i",
                device,
                "-t",
                "0.1",  # Record for 0.1 seconds
                "-ar",
                str(self.sample_rate),
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                test_file,
            ]

            test_result = subprocess.run(
                test_command, capture_output=True, text=True, timeout=5
            )

            if test_result.returncode == 0 and os.path.exists(test_file):
                file_size = os.path.getsize(test_file)
                self.logger.debug(
                    f"Device '{device}' test successful - created {file_size} byte test file"
                )
                os.remove(test_file)
                return True
            else:
                # Parse the error to provide better diagnostics
                error_msg = test_result.stderr.lower()
                if "device or resource busy" in error_msg:
                    self.logger.warning(
                        f"Device '{device}' is busy (being used by another process)"
                    )
                elif "no such file or directory" in error_msg:
                    self.logger.debug(f"Device '{device}' does not exist")
                elif "input/output error" in error_msg:
                    self.logger.debug(
                        f"Device '{device}' has I/O error (may be busy or misconfigured)"
                    )
                elif "permission denied" in error_msg:
                    self.logger.warning(
                        f"Device '{device}' permission denied (check audio group membership)"
                    )
                else:
                    self.logger.debug(
                        f"Device '{device}' test failed: {test_result.stderr.strip()}"
                    )
                return False

        except subprocess.TimeoutExpired:
            self.logger.debug(f"Device '{device}' test timed out")
            return False
        except Exception as e:
            self.logger.debug(f"Device '{device}' test error: {e}")
            return False

    def start_recording(self):
        """Start recording with enhanced processing and pre-record buffer."""
        if self.recording:
            # Check if we have a stale recording state
            if self.recording_start_time:
                stale_duration = (
                    datetime.datetime.now() - self.recording_start_time
                ).total_seconds()
                if stale_duration > self.max_recording_duration:
                    self.logger.warning(
                        f"Detected stale recording state ({stale_duration:.1f}s), cleaning up"
                    )
                    self.stop_recording()
                else:
                    self.logger.debug(
                        "Recording already in progress, ignoring start request"
                    )
                    return
            else:
                self.logger.warning(
                    "Recording flag set but no start time, resetting state"
                )
                self.recording = False

        os.makedirs("/tmp", exist_ok=True)
        timestamp = datetime.datetime.now().timestamp()

        # First, save the pre-record buffer
        pre_record_filename = f"pre_record_{timestamp:.3f}.wav"
        pre_record_filepath = os.path.join("/tmp", pre_record_filename)

        # Save buffered audio to catch the beginning of transmission
        if self._save_pre_record_buffer(pre_record_filepath):
            self.logger.info(f"Pre-record buffer saved: {pre_record_filepath}")

        # Create main recording file
        filename = f"recording_{timestamp:.3f}.wav"
        self.recording_filepath = os.path.join("/tmp", filename)
        self.recording_start_time = datetime.datetime.now()

        # Initialize recording state first (before starting FFmpeg)
        self.recording = True
        self.silence_counter = 0

        # Store pre-record filepath for later merging
        self.pre_record_filepath = (
            pre_record_filepath if os.path.exists(pre_record_filepath) else None
        )

        # Use a different approach: instead of starting a separate FFmpeg process,
        # we'll collect the audio data from the streaming process and save it
        # This avoids the audio device conflict issue
        self.recording_buffer = []  # Buffer to collect audio chunks during recording

        self.bus.publish(RxRecordingStartedEvent())
        self.logger.info(
            f"Enhanced recording started (streaming mode): {self.recording_filepath}"
        )
        if self.pre_record_filepath:
            self.logger.info(
                f"Pre-record buffer will be merged: {self.pre_record_filepath}"
            )

    def stop_recording(self):
        """Stop recording and apply advanced processing."""
        if not self.recording or self.recording_start_time is None:
            self.logger.debug("stop_recording called but not recording")
            return

        recording_duration = (
            datetime.datetime.now() - self.recording_start_time
        ).total_seconds()

        self.logger.info(f"Stopping recording after {recording_duration:.3f}s")

        # Save the collected audio data to file
        if hasattr(self, "recording_buffer") and self.recording_buffer:
            try:
                # Combine all recorded chunks
                combined_data = np.concatenate(self.recording_buffer)
                # Save as WAV file
                wavfile.write(self.recording_filepath, self.sample_rate, combined_data)
                self.logger.info(
                    f"Saved {len(combined_data)} samples to {self.recording_filepath}"
                )
            except Exception as e:
                self.logger.error(f"Failed to save recording buffer: {e}")
                self.recording = False
                self.silence_counter = 0
                self.bus.publish(RxRecordingEndedEvent())
                return

        self.recording = False
        self.silence_counter = 0  # Reset silence counter
        self.bus.publish(RxRecordingEndedEvent())

        # Give a moment for file operations to complete
        time.sleep(0.1)

        # Check if recording file actually exists
        if not self.recording_filepath or not os.path.exists(self.recording_filepath):
            self.logger.error(
                f"Recording file does not exist: {self.recording_filepath}"
            )
            return

        # Check file size to ensure we have actual audio data
        try:
            file_size = os.path.getsize(self.recording_filepath)
            self.logger.debug(f"Recording file size: {file_size} bytes")
            if file_size < 1024:  # Less than 1KB is probably an empty or corrupt file
                self.logger.warning(
                    f"Recording file is too small ({file_size} bytes), likely corrupt"
                )
                os.remove(self.recording_filepath)
                return
        except OSError as e:
            self.logger.error(f"Cannot access recording file: {e}")
            return

        # Calculate trimmed duration
        trimmed_duration = max(0, recording_duration - self.silence_duration)

        self.logger.info(
            f"Recording stopped - Duration: {recording_duration:.3f}s, Trimmed: {trimmed_duration:.3f}s, Min required: {self.minimal_recording_duration:.3f}s"
        )

        if trimmed_duration > self.minimal_recording_duration:
            self._process_recording(trimmed_duration)
        else:
            self.logger.info(
                f"Recording too short ({trimmed_duration:.3f}s), discarded"
            )
            if self.recording_filepath and os.path.exists(self.recording_filepath):
                os.remove(self.recording_filepath)

    @cpu_intensive
    def _process_recording(self, duration: float):
        """Apply advanced processing to recorded audio with pre-record buffer merge."""
        if not self.recording_filepath:
            return

        # Double-check that the recording file exists
        if not os.path.exists(self.recording_filepath):
            self.logger.error(
                f"Recording file missing for processing: {self.recording_filepath}"
            )
            return

        try:
            # Step 1: Merge pre-record buffer with main recording if available
            merged_filepath = self.recording_filepath

            if self.pre_record_filepath and os.path.exists(self.pre_record_filepath):
                # Create a merged file that includes the pre-record buffer
                merged_filepath = self.recording_filepath.replace(".wav", "_merged.wav")

                # Use FFmpeg to concatenate pre-record buffer and main recording
                concat_command = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    self.pre_record_filepath,
                    "-i",
                    self.recording_filepath,
                    "-filter_complex",
                    "[0:a][1:a]concat=n=2:v=0:a=1[a]",
                    "-map",
                    "[a]",
                    "-c:a",
                    "pcm_s16le",
                    merged_filepath,
                ]

                self.logger.debug(
                    f"Merging pre-record buffer with command: {' '.join(concat_command)}"
                )

                merge_result = subprocess.run(
                    concat_command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=15,
                )

                if merge_result.returncode == 0:
                    self.logger.info(
                        f"Successfully merged pre-record buffer with main recording"
                    )
                    # Update duration to include pre-record buffer
                    try:
                        sample_rate, merged_audio_data = wavfile.read(merged_filepath)
                        total_duration = len(merged_audio_data) / sample_rate
                        self.logger.info(
                            f"Total merged duration: {total_duration:.3f}s (was {duration:.3f}s)"
                        )
                        duration = total_duration
                    except Exception as e:
                        self.logger.warning(f"Could not read merged file duration: {e}")
                        duration += self.pre_record_duration  # Estimate
                else:
                    self.logger.error(
                        f"Failed to merge pre-record buffer: {merge_result.stderr}"
                    )
                    # Fall back to using original recording without pre-record buffer
                    merged_filepath = self.recording_filepath

                # Clean up pre-record buffer file
                try:
                    os.remove(self.pre_record_filepath)
                except:
                    pass

            processed_filepath = merged_filepath.replace(".wav", "_processed.ogg")

            # Advanced FFmpeg filter chain for radio processing
            filter_complex = (
                # Note: We don't trim the beginning anymore since we want to keep the pre-record buffer
                "[0:a]"
                # 1. High-pass filter to remove low-frequency noise
                "highpass=f=300,"
                # 2. De-emphasis (radio typically has pre-emphasis)
                "treble=g=-6:f=1000:width_type=h:width=1000,"
                # 3. Bandpass filter for voice clarity
                "lowpass=f=3400,"
                # 4. Dynamic range compression for consistent levels
                "compand=attacks=0.01:decays=0.5:points=-70/-70|-60/-50|-30/-20|-10/-10:soft-knee=6:gain=0:volume=-20,"
                # 5. Noise reduction (gentle)
                "afftdn=nr=10:nf=-50:tn=1,"
                # 6. Final limiter to prevent clipping
                "alimiter=level_in=1:level_out=1:limit=-1dB:attack=5:release=50[a]"
            )

            command = [
                "ffmpeg",
                "-y",
                "-i",
                merged_filepath,
                "-filter_complex",
                filter_complex,
                "-map",
                "[a]",
                "-ar",
                str(self.sample_rate),
                "-c:a",
                "libopus",
                "-b:a",
                "64k",  # Good quality for voice
                processed_filepath,
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                # Step 1: Analyze the processed recording for voice content
                is_voice, analysis = self.voice_detector.analyze_recording(
                    processed_filepath
                )

                self.logger.info(
                    f"Voice analysis - "
                    f"Contains voice: {is_voice}, "
                    f"Energy: {analysis.get('avg_energy_db', 0):.1f}dB, "
                    f"Voice ratio: {analysis.get('avg_voice_ratio', 0):.3f}, "
                    f"Voice quality: {analysis.get('avg_voice_quality', 0):.3f}, "
                    f"Consistency: {analysis.get('voice_consistency', 0):.1%}, "
                    f"Sustained: {analysis.get('sustained_energy_ratio', 0):.1%}, "
                    f"Spectral stability: {analysis.get('spectral_stability', 0):.3f}, "
                    f"Centroid: {analysis.get('avg_spectral_centroid', 0):.0f}Hz"
                )

                if is_voice:
                    # Voice detected - publish the processed recording
                    self.bus.publish(
                        RxRecordingCompleteEvent(filepath=processed_filepath)
                    )
                    self.logger.info(
                        f"Voice recording completed and sent: {processed_filepath}"
                    )
                else:
                    # No voice detected - discard the recording
                    self.logger.info(
                        f"No voice detected, discarding recording: {processed_filepath}"
                    )
                    if os.path.exists(processed_filepath):
                        os.remove(processed_filepath)
            else:
                self.logger.error(f"FFmpeg processing failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            self.logger.error("Recording processing timeout")
        except Exception as e:
            self.logger.error(f"Error processing recording: {e}")
        finally:
            # Clean up recording files
            if os.path.exists(self.recording_filepath):
                os.remove(self.recording_filepath)

            # Clean up merged file if it was created
            merged_filepath = self.recording_filepath.replace(".wav", "_merged.wav")
            if merged_filepath != self.recording_filepath and os.path.exists(
                merged_filepath
            ):
                os.remove(merged_filepath)

    def _process_audio_chunk_immediate(self, audio_chunk: np.ndarray):
        """
        Process audio chunk with immediate response (synchronous squelch processing).
        This method prioritizes speed over CPU efficiency for immediate squelch response.
        """
        start_time = time.time()
        try:
            # Always add to pre-record buffer (unless we're already recording)
            if not self.recording:
                self._add_to_pre_record_buffer(audio_chunk)

            # If recording, collect audio data
            if self.recording and hasattr(self, "recording_buffer"):
                self.recording_buffer.append(audio_chunk.copy())

            # Call squelch processing directly (synchronously) for immediate response
            # This bypasses the @cpu_intensive threading to eliminate latency
            squelch_open, stats = self._immediate_squelch_process(audio_chunk)

            # Handle squelch state changes immediately
            current_time = time.time()

            if squelch_open and not self.recording:
                # Squelch opened - start recording immediately (no delay needed now)
                self.start_recording()
                self.squelch_open_time = None
                self.logger.info("Recording started immediately on squelch open")

            elif not squelch_open:
                # Squelch closed
                if self.squelch_open_time is not None:
                    self.logger.debug("Squelch closed before recording threshold")
                self.squelch_open_time = None

                if self.recording:
                    self.silence_counter += 1
                    chunks_for_silence = self.silence_duration * (
                        self.sample_rate / self.chunk_size
                    )

                    self.logger.debug(
                        f"Silence counter: {self.silence_counter}/{chunks_for_silence:.0f}"
                    )

                    if self.silence_counter >= chunks_for_silence:
                        self.logger.info("Stopping recording due to silence timeout")
                        self.stop_recording()
            else:
                # Squelch is open and recording
                self.silence_counter = 0

                # Check for recording timeout
                if self.recording and self.recording_start_time:
                    recording_duration = (
                        datetime.datetime.now() - self.recording_start_time
                    ).total_seconds()

                    if recording_duration > self.max_recording_duration:
                        self.logger.warning(
                            f"Recording timeout after {recording_duration:.1f}s, force stopping"
                        )
                        self.stop_recording()

            # Track processing performance
            process_time = time.time() - start_time
            self.chunk_process_times.append(process_time)
            if len(self.chunk_process_times) > self.max_process_time_samples:
                self.chunk_process_times.pop(0)

            # Schedule statistics publishing in the bus
            if current_time - self.last_stats_publish_time >= self.stats_interval:
                # Add performance stats
                avg_process_time = sum(self.chunk_process_times) / len(
                    self.chunk_process_times
                )
                max_process_time = max(self.chunk_process_times)
                stats["avg_process_time_ms"] = avg_process_time * 1000
                stats["max_process_time_ms"] = max_process_time * 1000

                # Periodic health check for recording
                self._periodic_recording_health_check()

                # Publish stats directly (no async needed)
                self._publish_enhanced_stats(stats)
                self.last_stats_publish_time = current_time

        except Exception as e:
            self.logger.error(f"Error in immediate audio chunk processing: {e}")

    def _periodic_recording_health_check(self):
        """Periodic health check for recording state and FFmpeg process."""
        try:
            # Check recording timeout and health
            if self.recording:
                if self.recording_start_time:
                    recording_duration = (
                        datetime.datetime.now() - self.recording_start_time
                    ).total_seconds()

                    if recording_duration > self.max_recording_duration:
                        self.logger.warning(
                            f"Periodic check: Recording timeout after {recording_duration:.1f}s, force stopping"
                        )
                        self.stop_recording()
                        return

                # Log recording status
                if self.recording_start_time:
                    recording_duration = (
                        datetime.datetime.now() - self.recording_start_time
                    ).total_seconds()
                    buffer_size = len(getattr(self, "recording_buffer", []))
                    self.logger.debug(
                        f"Recording health check: {recording_duration:.1f}s/{self.max_recording_duration:.0f}s, "
                        f"silence_counter: {self.silence_counter}, buffer_chunks: {buffer_size}"
                    )

            # Check audio streaming process health (runs even when not recording)
            if self.streaming_process:
                if self.streaming_process.poll() is not None:
                    self.logger.error("Periodic check: FFmpeg streaming process died")
                    # Don't try to restart here, let the main loop handle it
                    return

                # Check if we're receiving audio data
                current_time = time.time()
                if (
                    self.last_chunk_time > 0
                ):  # Only check if we've received at least one chunk
                    time_since_last_chunk = current_time - self.last_chunk_time
                    if time_since_last_chunk > self.streaming_timeout:
                        self.logger.warning(
                            f"No audio data received for {time_since_last_chunk:.1f}s, streaming may be hung"
                        )
                        # Log additional debug info
                        if self.streaming_process:
                            self.logger.warning(
                                f"Streaming process PID: {self.streaming_process.pid}, poll: {self.streaming_process.poll()}"
                            )

        except Exception as e:
            self.logger.error(f"Error in periodic recording health check: {e}")

    def _immediate_squelch_process(self, audio_chunk: np.ndarray) -> Tuple[bool, dict]:
        """
        Immediate squelch processing without threading for minimal latency.
        This is a simplified version of the squelch processing optimized for speed.
        """
        try:
            # Basic level analysis (fast)
            rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
            current_db = 20 * np.log10(rms / 32767.0 + 1e-6)

            # Update noise floor (don't update if currently open)
            noise_floor_db = self.squelch.noise_floor.update(
                current_db, self.squelch.is_open
            )
            level_margin_db = current_db - noise_floor_db

            # Simplified spectral analysis for speed
            if (
                len(audio_chunk) >= 256
            ):  # Only do spectral analysis if we have enough samples
                # Use a smaller FFT for speed
                fft_chunk = audio_chunk[:256] if len(audio_chunk) > 256 else audio_chunk
                voice_energy, total_energy, spectral_centroid, voice_quality = (
                    self.squelch.spectral_analyzer.analyze_spectrum(fft_chunk)
                )
            else:
                # Skip spectral analysis for very small chunks
                voice_energy = 0
                total_energy = 1
                spectral_centroid = 1000  # Assume reasonable default
                voice_quality = 0

            # Simplified scoring for immediate response
            level_score = max(
                0, min(1, (level_margin_db - 3) / 15)
            )  # Smooth 3-18dB range

            if total_energy > 0:
                voice_ratio_score = min(
                    1, voice_energy / total_energy / self.squelch.voice_energy_ratio
                )
            else:
                voice_ratio_score = 0

            # Simplified spectral centroid score
            if (
                self.squelch.spectral_centroid_min
                <= spectral_centroid
                <= self.squelch.spectral_centroid_max
            ):
                centroid_score = 1.0
            else:
                centroid_score = 0.5

            # Simplified voice quality score
            quality_score = min(1.0, voice_quality / self.squelch.min_voice_quality)

            # Absolute level check
            if current_db < self.squelch.min_absolute_level:
                level_score = 0

            # Simplified scoring (prioritize level for immediate response)
            combined_score = (
                0.5 * level_score  # Higher weight on signal strength for fast response
                + 0.2 * voice_ratio_score  # Reduced weight on complex analysis
                + 0.15 * centroid_score
                + 0.15 * quality_score
            )

            # Apply hysteresis
            threshold = (
                self.squelch.close_threshold
                if self.squelch.is_open
                else self.squelch.open_threshold
            )
            self.squelch.is_open = combined_score >= threshold

            stats = {
                "current_db": current_db,
                "noise_floor_db": noise_floor_db,
                "level_margin_db": level_margin_db,
                "voice_energy": voice_energy,
                "total_energy": total_energy,
                "spectral_centroid": spectral_centroid,
                "voice_quality": voice_quality,
                "level_score": level_score,
                "voice_ratio_score": voice_ratio_score,
                "centroid_score": centroid_score,
                "quality_score": quality_score,
                "combined_score": combined_score,
                "squelch_open": self.squelch.is_open,
            }

            return self.squelch.is_open, stats

        except Exception as e:
            self.logger.error(f"Error in immediate squelch processing: {e}")
            # Return safe defaults
            return False, {
                "current_db": -60.0,
                "noise_floor_db": -60.0,
                "level_margin_db": 0.0,
                "voice_energy": 0,
                "total_energy": 1,
                "spectral_centroid": 1000,
                "voice_quality": 0,
                "level_score": 0,
                "voice_ratio_score": 0,
                "centroid_score": 0,
                "quality_score": 0,
                "combined_score": 0,
                "squelch_open": False,
            }

    def _process_audio_chunk(self, audio_chunk: np.ndarray):
        """
        Process audio chunk using the threaded bus system.
        The @cpu_intensive decorator on squelch.process() handles threading automatically.
        """
        try:
            # Always add to pre-record buffer (unless we're already recording)
            if not self.recording:
                self._add_to_pre_record_buffer(audio_chunk)

            # The squelch.process method is marked with @cpu_intensive, so it will
            # automatically run in a separate thread via the bus system
            squelch_open, stats = self.squelch.process(audio_chunk)

            # Handle squelch state changes (fast operations, synchronous)
            current_time = time.time()

            if squelch_open and not self.recording:
                # Squelch opened - start recording immediately (no delay needed now)
                self.start_recording()
                self.squelch_open_time = None

            elif not squelch_open:
                # Squelch closed
                self.squelch_open_time = None

                if self.recording:
                    self.silence_counter += 1
                    chunks_for_silence = self.silence_duration * (
                        self.sample_rate / self.chunk_size
                    )

                    self.logger.debug(
                        f"Silence counter: {self.silence_counter}/{chunks_for_silence:.0f}"
                    )

                    if self.silence_counter >= chunks_for_silence:
                        self.logger.info("Stopping recording due to silence timeout")
                        self.stop_recording()
            else:
                # Squelch is open and recording
                self.silence_counter = 0

                # Check for recording timeout and FFmpeg health
                if self.recording and self.recording_start_time:
                    recording_duration = (
                        datetime.datetime.now() - self.recording_start_time
                    ).total_seconds()

                    if recording_duration > self.max_recording_duration:
                        self.logger.warning(
                            f"Recording timeout after {recording_duration:.1f}s, force stopping"
                        )
                        self.stop_recording()

                    # Check if FFmpeg process is still alive
                    if self.process and self.process.poll() is not None:
                        self.logger.error(
                            "FFmpeg recording process died unexpectedly, stopping recording"
                        )
                        self.stop_recording()

            # Schedule statistics publishing in the bus
            if current_time - self.last_stats_publish_time >= self.stats_interval:
                # Periodic health check for recording
                self._periodic_recording_health_check()

                # Publish stats directly (no async needed)
                self._publish_enhanced_stats(stats)
                self.last_stats_publish_time = current_time

        except Exception as e:
            self.logger.error(f"Error in audio chunk processing: {e}")

    def process_audio_stream(self):
        """Enhanced audio stream processing with advanced squelch using FFmpeg streaming."""
        self.logger.info("Starting enhanced audio stream processing with FFmpeg")

        max_retries = 3
        retry_delay = 5.0  # seconds

        for attempt in range(max_retries):
            if not self.enabled:
                self.logger.info("Audio processing disabled, stopping")
                return

            self.logger.info(
                f"Audio stream attempt {attempt + 1}/{max_retries} with device: {self.audio_device}"
            )

            if self._process_audio_stream_once():
                # Successful run, don't retry unless it fails again
                break
            else:
                # Failed, try to find a different device for next attempt
                if attempt < max_retries - 1:
                    self.logger.warning(
                        f"Audio stream failed, retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)

                    # Try to find a working device for next attempt
                    self._test_audio_device()
                else:
                    self.logger.error("All audio stream attempts failed")

    def _process_audio_stream_once(self) -> bool:
        """Single attempt at audio stream processing. Returns True if successful."""
        # Use FFmpeg for streaming with minimal latency settings
        ffmpeg_cmd = [
            "ffmpeg",
            "-f",
            "alsa",
            "-i",
            self.audio_device,
            "-ac",
            "1",  # Mono
            "-ar",
            str(self.sample_rate),  # Sample rate
            "-acodec",
            "pcm_s16le",  # Direct PCM encoding
            "-fflags",
            "nobuffer",  # Disable buffering for low latency
            "-flags",
            "low_delay",  # Enable low delay mode
            "-strict",
            "experimental",
            "-f",
            "s16le",  # 16-bit little-endian PCM
            "-",  # Output to stdout
        ]

        process = None
        try:
            # Start FFmpeg process for continuous audio streaming with minimal buffering
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # Completely unbuffered
            )

            # Store reference for health monitoring
            self.streaming_process = process

            if not process.stdout:
                self.logger.error("Failed to get FFmpeg stdout stream")
                return False

            self.logger.info(
                f"FFmpeg low-latency audio streaming started with device: {self.audio_device}"
            )

            # Use smaller chunks for immediate response
            bytes_per_sample = 2  # 16-bit = 2 bytes
            # Reduce chunk size for faster response (was self.chunk_size)
            small_chunk_size = min(
                256, self.chunk_size
            )  # Use 256 samples max for low latency
            chunk_bytes = small_chunk_size * bytes_per_sample

            # Track successful operation
            successful_chunks = 0
            min_successful_chunks = 10

            # Buffer for accumulating partial data
            audio_buffer = bytearray()

            try:
                while True:
                    if not self.enabled:
                        self.logger.info("Audio processing disabled, stopping stream")
                        return successful_chunks >= min_successful_chunks

                    # Periodic safety check for recording timeout
                    if self.recording and self.recording_start_time:
                        recording_duration = (
                            datetime.datetime.now() - self.recording_start_time
                        ).total_seconds()
                        if (
                            recording_duration > self.max_recording_duration * 1.5
                        ):  # 1.5x timeout for emergency stop
                            self.logger.error(
                                f"Emergency stop: recording has been running for {recording_duration:.1f}s"
                            )
                            self.stop_recording()

                    try:
                        # Check if FFmpeg process is still running
                        if process.poll() is not None:
                            # Process has terminated, try to get error info
                            _, stderr = process.communicate()
                            error_msg = stderr.decode() if stderr else "Unknown error"
                            self.logger.error(f"FFmpeg process terminated: {error_msg}")
                            return successful_chunks >= min_successful_chunks

                        # Use non-blocking read with select for immediate response
                        import select

                        # Check if data is available for reading (timeout = 0.001s for immediate response)
                        ready, _, _ = select.select([process.stdout], [], [], 0.001)

                        if not ready:
                            # No data available, continue immediately (don't sleep)
                            continue

                        # Read available data (may be less than chunk_bytes)
                        raw_data = process.stdout.read(chunk_bytes)

                        if not raw_data:
                            # No data available, FFmpeg might have stopped
                            continue

                        # Add to buffer
                        audio_buffer.extend(raw_data)

                        # Process complete chunks immediately when available
                        while len(audio_buffer) >= chunk_bytes:
                            # Extract one complete chunk
                            chunk_data = audio_buffer[:chunk_bytes]
                            audio_buffer = audio_buffer[chunk_bytes:]

                            # Convert raw bytes to numpy array
                            audio_chunk = np.frombuffer(chunk_data, dtype=np.int16)

                            # Process audio chunk with immediate response
                            try:
                                self._process_audio_chunk_immediate(audio_chunk)
                                successful_chunks += 1
                                # Update last chunk time for health monitoring
                                self.last_chunk_time = time.time()
                            except Exception as processing_error:
                                # If processing fails, log but continue with audio stream
                                self.logger.debug(
                                    f"Audio processing error: {processing_error}"
                                )
                                successful_chunks += 1  # Still count as successful
                                # Still update chunk time even on processing errors
                                self.last_chunk_time = time.time()

                    except Exception as e:
                        self.logger.error(f"Error reading audio chunk: {e}")
                        # Don't sleep on errors, continue immediately

            finally:
                pass  # No async processing cleanup needed anymore

        except Exception as e:
            self.logger.error(f"Fatal error in audio stream processing: {e}")
            return False
        finally:
            # Clean up FFmpeg process
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.logger.warning("FFmpeg termination timeout, force killing")
                    process.kill()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.logger.error("Could not kill FFmpeg process")
                except Exception as e:
                    self.logger.error(f"Error terminating FFmpeg process: {e}")

            # Clear streaming process reference
            self.streaming_process = None
            self.logger.info("Audio stream processing stopped")

    def _publish_enhanced_stats(self, stats: dict):
        """Publish enhanced statistics including spectral analysis and performance metrics."""
        # Publish the new unified audio stats event
        self.bus.publish(
            RxAudioStatsEvent(
                current_db=float(stats["current_db"]),
                noise_floor_db=float(stats["noise_floor_db"]),
                level_margin_db=float(stats["level_margin_db"]),
                squelch_open=bool(stats["squelch_open"]),
                voice_ratio_score=float(stats["voice_ratio_score"]),
                voice_quality=float(stats["voice_quality"]),
                spectral_centroid=float(stats["spectral_centroid"]),
                avg_process_time_ms=float(stats.get("avg_process_time_ms", 0)),
                max_process_time_ms=float(stats.get("max_process_time_ms", 0)),
                voice_energy=float(stats.get("voice_energy", 0)),
                total_energy=float(stats.get("total_energy", 0)),
                level_score=float(stats.get("level_score", 0)),
                centroid_score=float(stats.get("centroid_score", 0)),
                quality_score=float(stats.get("quality_score", 0)),
                combined_score=float(stats.get("combined_score", 0)),
            )
        )

        # Log detailed stats for debugging/monitoring including performance
        performance_info = ""
        if "avg_process_time_ms" in stats and "max_process_time_ms" in stats:
            performance_info = f", Avg process: {stats['avg_process_time_ms']:.2f}ms, Max: {stats['max_process_time_ms']:.2f}ms"

        self.logger.info(
            f"Audio Stats - "
            f"Level: {stats['current_db']:.1f}dB, "
            f"Noise: {stats['noise_floor_db']:.1f}dB, "
            f"Margin: {stats['level_margin_db']:.1f}dB, "
            f"Squelch: {'OPEN' if stats['squelch_open'] else 'CLOSED'}, "
            f"Voice ratio: {stats['voice_ratio_score']:.2f}, "
            f"Voice quality: {stats['voice_quality']:.2f}, "
            f"Spectral centroid: {stats['spectral_centroid']:.0f}Hz"
            f"{performance_info}"
        )

    def start(self):
        """Start the enhanced audio processing worker."""
        self.logger.info("Enhanced RxListenWorker starting...")

        # Start the main audio stream processing in a separate thread
        # This allows the sync FFmpeg reading loop to run without blocking
        self.audio_thread = threading.Thread(
            target=self.process_audio_stream, daemon=True
        )
        self.audio_thread.start()

    def stop(self):
        """Stop the audio processing worker and clean up resources."""
        self.logger.info("Stopping Enhanced RxListenWorker...")

        self.enabled = False

        # Stop any ongoing recording
        if self.recording:
            self.stop_recording()

        # Stop streaming FFmpeg process if it's running
        if self.streaming_process and self.streaming_process.poll() is None:
            self.logger.info("Terminating streaming FFmpeg process")
            try:
                self.streaming_process.terminate()
                self.streaming_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger.warning(
                    "Streaming FFmpeg termination timeout, force killing"
                )
                self.streaming_process.kill()
                try:
                    self.streaming_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.logger.error("Could not kill streaming FFmpeg process")
            except Exception as e:
                self.logger.error(f"Error terminating streaming FFmpeg process: {e}")
            finally:
                self.streaming_process = None

        self.logger.info("Enhanced RxListenWorker stopped")
