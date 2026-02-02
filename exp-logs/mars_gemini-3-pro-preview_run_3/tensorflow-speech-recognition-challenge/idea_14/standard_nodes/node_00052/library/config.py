import os
import torch
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class PathConfig:
    """Defines file paths for data, metadata, and outputs."""

    input_root: str = "./input"
    train_audio_dir: str = os.path.join(input_root, "train", "audio")
    test_audio_dir: str = os.path.join(input_root, "test", "audio")

    metadata_dir: str = "./metadata"
    train_csv: str = os.path.join(metadata_dir, "train.csv")
    val_csv: str = os.path.join(metadata_dir, "val.csv")
    test_csv: str = os.path.join(metadata_dir, "test.csv")

    noise_dir: str = os.path.join(train_audio_dir, "_background_noise_")

    working_dir: str = "./working/idea_14"
    submission_path: str = "./submission/submission.csv"

    def __post_init__(self):
        """Ensure necessary output directories exist."""
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.submission_path), exist_ok=True)


@dataclass
class AudioConfig:
    """Defines raw audio processing parameters."""

    sample_rate: int = 16000
    duration: float = 1.0
    n_samples: int = 16000  # sample_rate * duration


@dataclass
class MelConfig:
    """Defines Multi-Resolution Log-Mel Spectrogram parameters."""

    n_mels: int = 128
    f_min: int = 20
    f_max: int = 8000
    hop_length: int = 160  # 10ms at 16kHz

    # Multi-resolution settings:
    # Short (20ms/320 samples), Medium (40ms/640 samples), Long (60ms/960 samples)
    # n_ffts are chosen as the next power of 2 for the respective window lengths
    win_lengths: List[int] = field(default_factory=lambda: [320, 640, 960])
    n_ffts: List[int] = field(default_factory=lambda: [512, 1024, 2048])


@dataclass
class ModelConfig:
    """Defines the ResNeSt-CRNN architecture parameters."""

    backbone: str = "resnest50d"
    pretrained: bool = True
    in_channels: int = 3
    num_classes: int = 12

    # Neck (BiGRU) and Head (Attention) parameters
    hidden_size: int = 512
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.3


@dataclass
class TrainConfig:
    """Defines training hyperparameters and augmentation settings."""

    seed: int = 42
    batch_size: int = 64
    num_epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    early_stopping_patience: int = 10

    # Augmentation Parameters
    noise_prob: float = 0.5
    noise_snr_min: float = 0.0
    noise_snr_max: float = 15.0

    spec_aug_prob: float = 0.5
    freq_mask_param: int = 20
    time_mask_param: int = 20  # Approx <20% of time steps

    # Debugging
    debug: bool = False
    debug_sample_size: int = 500


# Global Constants for Label Mapping
LABELS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "silence",
    "unknown",
]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}
IDX_TO_LABEL = {i: l for i, l in enumerate(LABELS)}
