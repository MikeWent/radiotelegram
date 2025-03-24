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
class RxNoiseFloorStatsEvent:
    ambient_db: float
    current_db: float
    delta_db: float


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
