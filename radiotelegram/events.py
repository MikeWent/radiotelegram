from dataclasses import dataclass


# Recording
@dataclass
class RxRecordingStartedEvent:
    pass


@dataclass
class RxRecordingEndedEvent:
    pass


@dataclass
class RxRecordingCompleteEvent:
    filepath: str


@dataclass
class RxAudioStatsEvent:
    """Unified audio statistics event including all metrics and squelch status."""

    # Basic audio levels
    current_db: float
    noise_floor_db: float
    level_margin_db: float

    # Squelch status
    squelch_open: bool

    # TEN-VAD metrics
    vad_probability: float
    vad_flag: int

    # Scoring
    level_score: float = 0.0
    combined_score: float = 0.0

    # Squelch thresholds (for tuning visibility)
    open_threshold: float = 0.0
    close_threshold: float = 0.0
    active_threshold: float = 0.0
    min_absolute_level: float = 0.0

    # Noise floor details
    fast_noise_floor_db: float = 0.0
    slow_noise_floor_db: float = 0.0

    # Recording state
    is_recording: bool = False

    # Optional performance metrics
    avg_process_time_ms: float = 0.0
    max_process_time_ms: float = 0.0


# Playback
@dataclass
class TxMessagePlaybackStartedEvent:
    pass


@dataclass
class TxMessagePlaybackEndedEvent:
    pass


# Telegram
@dataclass
class TelegramVoiceMessageDownloadedEvent:
    filepath: str
