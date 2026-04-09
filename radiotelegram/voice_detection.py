"""Voice detection using TEN-VAD for post-recording analysis."""

import os
import subprocess

import numpy as np
from scipy.io import wavfile
from ten_vad import TenVad

_TEN_VAD_SAMPLE_RATE = 16000


class VoiceDetector:
    """Analyzes recorded audio to determine if it contains voice."""

    def __init__(
        self,
        sample_rate=48000,
        hop_size=256,
        threshold=0.5,
        min_voice_ratio=0.01,
        min_max_probability=0.55,
        min_analysis_duration=0.5,
        min_avg_probability=0.200,
    ):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.threshold = threshold
        self.min_voice_ratio = min_voice_ratio
        self.min_max_probability = min_max_probability
        self.min_analysis_duration = min_analysis_duration
        self.min_avg_probability = min_avg_probability
        self.vad = TenVad(hop_size=hop_size, threshold=threshold)

    def analyze_recording(self, filepath):
        try:
            temp_wav = filepath + "_vad.wav"
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    filepath,
                    "-ar",
                    str(_TEN_VAD_SAMPLE_RATE),
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    temp_wav,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return False, {"error": "Failed to convert audio"}

            try:
                _, audio_data = wavfile.read(temp_wav)
            finally:
                os.remove(temp_wav)

            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]

            duration = len(audio_data) / _TEN_VAD_SAMPLE_RATE
            if duration < self.min_analysis_duration:
                return False, {"error": "Recording too short for analysis"}

            total_frames = 0
            voice_frames = 0
            probabilities = []

            for i in range(0, len(audio_data) - self.hop_size + 1, self.hop_size):
                frame = audio_data[i : i + self.hop_size]
                probability, flag = self.vad.process(frame)
                probabilities.append(probability)
                if flag == 1:
                    voice_frames += 1
                total_frames += 1

            if total_frames == 0:
                return False, {"error": "No audio frames to analyze"}

            voice_ratio = voice_frames / total_frames
            probs = np.array(probabilities)
            avg_probability = float(np.mean(probs))
            max_probability = float(np.max(probs))
            std_probability = float(np.std(probs))
            p25, p50, p75, p90, p95 = (
                float(v) for v in np.percentile(probs, [25, 50, 75, 90, 95])
            )

            is_voice = (
                voice_ratio > self.min_voice_ratio
                and max_probability > self.min_max_probability
                and avg_probability > self.min_avg_probability
            )

            return bool(is_voice), {
                "duration": duration,
                "voice_ratio": voice_ratio,
                "avg_probability": avg_probability,
                "max_probability": max_probability,
                "std_probability": std_probability,
                "p25": p25,
                "p50": p50,
                "p75": p75,
                "p90": p90,
                "p95": p95,
                "voice_frames": voice_frames,
                "total_frames": total_frames,
                "is_voice": is_voice,
                "threshold": self.threshold,
                "min_voice_ratio": self.min_voice_ratio,
                "min_max_probability": self.min_max_probability,
                "min_avg_probability": self.min_avg_probability,
            }
        except Exception as e:
            return False, {"error": f"Analysis failed: {str(e)}"}
