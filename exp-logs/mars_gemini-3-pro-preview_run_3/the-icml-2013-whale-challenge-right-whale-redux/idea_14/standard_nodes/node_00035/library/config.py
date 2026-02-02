import os
import torch
from dataclasses import dataclass


@dataclass
class AudioConfig:
    """
    Configuration for Audio Processing and Spectrogram Generation.
    Parameters chosen to match EfficientNetV2-Medium capacity and signal characteristics.
    """

    sample_rate: int = 2000
    n_fft: int = 1024
    hop_length: int = 20  # Approx 10ms at 2kHz
    n_mels: int = 384  # Matches EfficientNetV2-M compound scaling
    fmin: int = 0
    fmax: int = None  # None implies Nyquist (sample_rate / 2)


@dataclass
class ModelConfig:
    """
    Configuration for the Neural Network Architecture.
    Uses EfficientNetV2-Medium with GeM pooling.
    """

    backbone: str = "tf_efficientnetv2_m"
    pretrained: bool = True
    num_classes: int = 1
    pool_type: str = "gem"
    drop_path_rate: float = 0.2
    dropout_rate: float = 0.3
    in_channels: int = 1


@dataclass
class TrainConfig:
    """
    Configuration for Training Hyperparameters and Strategies.
    Includes settings for the Noisy Student pipeline and debugging.
    """

    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 1e-2
    mixup_alpha: float = 0.4  # Calibrated for signal structure preservation
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    early_stopping_patience: int = 5

    # Debugging flags
    debug: bool = False
    debug_sample_size: int = 200  # Number of samples to use when debug=True


@dataclass
class PathConfig:
    """
    Configuration for File Paths and Directories.
    """

    input_dir: str = "./input"
    train_dir: str = os.path.join(input_dir, "train2")
    test_dir: str = os.path.join(input_dir, "test2")

    metadata_dir: str = "./metadata"
    train_meta: str = os.path.join(metadata_dir, "train.csv")
    val_meta: str = os.path.join(metadata_dir, "val.csv")
    test_meta: str = os.path.join(metadata_dir, "test.csv")

    working_dir: str = "./working/idea_14"
    cache_dir: str = os.path.join(working_dir, "cache")
    checkpoint_dir: str = os.path.join(working_dir, "checkpoints")
    submission_dir: str = "./submission"
    submission_path: str = os.path.join(submission_dir, "submission.csv")

    # Specific checkpoint paths for the pipeline
    teacher_checkpoint: str = os.path.join(checkpoint_dir, "teacher_best.pth")
    student_checkpoint: str = os.path.join(checkpoint_dir, "student_best.pth")

    @classmethod
    def create_dirs(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.cache_dir, exist_ok=True)
        os.makedirs(cls.checkpoint_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)
