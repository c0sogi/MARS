import os
from dataclasses import dataclass, field
from typing import Set

# Ensure working directory exists
WORKING_DIR = "./working/idea_7"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


@dataclass
class AudioConfig:
    """Configuration for audio processing and spectrogram generation."""

    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 160
    n_mels: int = 128
    duration: float = 1.0  # seconds
    f_min: int = 0
    f_max: int = 8000  # Nyquist frequency

    @property
    def num_samples(self) -> int:
        return int(self.sample_rate * self.duration)


@dataclass
class PathConfig:
    """Configuration for file paths."""

    input_dir: str = "./input"
    train_audio_dir: str = os.path.join(input_dir, "train", "audio")
    test_audio_dir: str = os.path.join(input_dir, "test", "audio")
    background_noise_dir: str = os.path.join(train_audio_dir, "_background_noise_")

    metadata_dir: str = "./metadata"
    train_metadata: str = os.path.join(metadata_dir, "train.csv")
    val_metadata: str = os.path.join(metadata_dir, "val.csv")
    test_metadata: str = os.path.join(metadata_dir, "test.csv")

    working_dir: str = WORKING_DIR
    submission_path: str = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Artifact paths
    model_save_path: str = os.path.join(working_dir, "best_model.pth")
    label_encoder_path: str = os.path.join(working_dir, "label_encoder.npy")
    cache_path: str = os.path.join(working_dir, "dataset_cache.parquet")


@dataclass
class TrainConfig:
    """Configuration for training loop and hyperparameters."""

    seed: int = 42
    batch_size: int = 32
    epochs: int = 45
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2

    # Regularization
    mixup_alpha: float = 1.0
    spec_augment_time_mask: int = 30
    spec_augment_freq_mask: int = 20

    # Scheduler
    eta_min: float = 1e-6

    # System
    num_workers: int = 4
    device: str = "cuda"

    # Debugging / Development
    debug: bool = False
    debug_sample_size: int = 500
    early_stopping_patience: int = 10


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    backbone: str = "efficientnet_b2"
    pretrained: bool = True
    in_channels: int = 1
    drop_rate: float = 0.3
    drop_path_rate: float = 0.2


@dataclass
class LabelConfig:
    """Configuration for label handling and submission mapping."""

    # The 10 specific commands we need to predict
    target_labels: Set[str] = field(
        default_factory=lambda: {
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
        }
    )

    # Special labels
    silence_label: str = "silence"
    unknown_label: str = "unknown"

    # Balancing strategy
    target_upsample_count: int = 2000


# Instantiate configs for easy import
audio_config = AudioConfig()
path_config = PathConfig()
train_config = TrainConfig()
model_config = ModelConfig()
label_config = LabelConfig()
