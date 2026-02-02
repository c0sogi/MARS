import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PathConfig:
    """
    Defines file paths for the project.
    """

    input_dir: str = "./input"
    train_audio_dir: str = os.path.join(input_dir, "train", "audio")
    test_audio_dir: str = os.path.join(input_dir, "test", "audio")

    metadata_dir: str = "./metadata"
    train_metadata_path: str = os.path.join(metadata_dir, "train.csv")
    val_metadata_path: str = os.path.join(metadata_dir, "val.csv")
    test_metadata_path: str = os.path.join(metadata_dir, "test.csv")

    # Working directory for idea_13 as specified
    working_dir: str = "./working/idea_13"
    model_checkpoint_path: str = os.path.join(working_dir, "best_model.pth")
    last_checkpoint_path: str = os.path.join(working_dir, "last_checkpoint.pth")

    # Submission
    submission_dir: str = "./submission"
    submission_path: str = os.path.join(submission_dir, "submission.csv")

    def __post_init__(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)


@dataclass
class AudioConfig:
    """
    Configuration for Audio Signal Processing.
    Implements the High-Fidelity Signal Processing Paradigm.
    """

    sample_rate: int = 16000
    duration: int = 1  # seconds

    # Spectral Oversampling parameters
    n_fft: int = 1024  # Increased from default for better interpolation
    win_length: int = 400  # 25ms window
    hop_length: int = 160  # 10ms stride

    # Mel Spectrogram parameters
    n_mels: int = 128  # High resolution
    fmin: int = 20
    fmax: int = 8000  # Nyquist at 16k SR
    top_db: float = 80.0  # Log-mel dynamic range

    # Target Labels
    labels: List[str] = field(
        default_factory=lambda: [
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
    )

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    @property
    def label_to_idx(self) -> dict:
        return {label: idx for idx, label in enumerate(self.labels)}


@dataclass
class ModelConfig:
    """
    Configuration for the Neural Network Architecture.
    """

    model_name: str = "tf_efficientnetv2_b0"
    pretrained: bool = True
    in_channels: int = 1  # Spectrogram input
    drop_rate: float = 0.2
    drop_path_rate: float = 0.1

    # EMA Settings
    use_ema: bool = True
    ema_decay: float = 0.999


@dataclass
class TrainConfig:
    """
    Configuration for Training Hyperparameters.
    """

    seed: int = 42

    # Optimization
    batch_size: int = 32  # Small batch size for convergence
    epochs: int = 45  # Long schedule
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2  # AdamW default
    label_smoothing: float = 0.1

    # Scheduler
    scheduler_type: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Hardware
    num_workers: int = 4
    use_mixed_precision: bool = True

    # Early Stopping
    early_stopping_patience: int = 10

    # Augmentation
    mix_noise_prob: float = 0.5
    spec_augment_prob: float = 0.5
    time_mask_param: int = 20  # Conservative masking
    freq_mask_param: int = 10  # Conservative masking

    # Debugging
    debug: bool = False
    debug_sample_size: int = 1000  # Number of samples to use in debug mode
