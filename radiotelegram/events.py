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

    # Voice detection metrics
    voice_ratio_score: float
    voice_quality: float
    spectral_centroid: float

    # Optional performance metrics
    avg_process_time_ms: float = 0.0
    max_process_time_ms: float = 0.0

    # Additional spectral data
    voice_energy: float = 0.0
    total_energy: float = 0.0
    level_score: float = 0.0
    centroid_score: float = 0.0
    quality_score: float = 0.0
    combined_score: float = 0.0


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
