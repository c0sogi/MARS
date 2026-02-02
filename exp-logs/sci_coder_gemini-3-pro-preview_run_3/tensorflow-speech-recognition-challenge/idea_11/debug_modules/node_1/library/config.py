import os
import random
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PathConfig:
    """Defines input and output paths for the project."""

    input_dir: str = "./input"
    train_audio_dir: str = os.path.join(input_dir, "train", "audio")
    test_audio_dir: str = os.path.join(input_dir, "test", "audio")

    metadata_dir: str = "./metadata"
    train_meta: str = os.path.join(metadata_dir, "train.csv")
    val_meta: str = os.path.join(metadata_dir, "val.csv")
    test_meta: str = os.path.join(metadata_dir, "test.csv")

    # Working directory for Idea 11
    working_dir: str = "./working/idea_11"
    cache_dir: str = os.path.join(working_dir, "cache")
    model_save_path: str = os.path.join(working_dir, "best_model.pth")
    submission_dir: str = "./submission"
    submission_path: str = os.path.join(submission_dir, "submission.csv")

    def __post_init__(self):
        """Ensure necessary writeable directories exist."""
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)


@dataclass
class AudioConfig:
    """Defines audio processing and spectrogram generation parameters."""

    sample_rate: int = 16000
    duration: float = 1.0
    num_samples: int = 16000  # sample_rate * duration

    n_mels: int = 64
    f_min: int = 20
    f_max: int = 8000

    # Multi-Resolution STFT settings
    # We use a fixed hop_length to ensure all channels have the same temporal dimension.
    # 160 samples @ 16kHz = 10ms hop
    hop_length: int = 160

    # Window sizes for the 3 channels: Short (20ms), Medium (40ms), Long (60ms)
    # 20ms * 16000 = 320
    # 40ms * 16000 = 640
    # 60ms * 16000 = 960
    win_lengths: Tuple[int, int, int] = (320, 640, 960)

    # FFT size must be large enough for the largest window (960)
    n_fft: int = 2048


@dataclass
class ModelConfig:
    """Defines model architecture hyperparameters."""

    backbone: str = "skresnet34"
    pretrained: bool = True
    in_channels: int = 3  # Corresponds to the 3 multi-res windows
    num_classes: int = 12

    # Feature Aggregation
    use_hierarchical: bool = True  # Fuse layers 2, 3, 4

    # Recurrent Neck
    gru_hidden_size: int = 256
    gru_layers: int = 2
    gru_dropout: float = 0.3

    # Attention Head
    attention_heads: int = 4
    dropout: float = 0.3


@dataclass
class TrainConfig:
    """Defines training loop hyperparameters."""

    seed: int = 42
    batch_size: int = 64
    epochs: int = 25

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2

    # Scheduler (Cosine Annealing)
    T_max: int = 25
    eta_min: float = 1e-6

    # Early Stopping
    patience: int = 5

    # Hardware
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    """Sets fixed random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Initialize global config instances
path_cfg = PathConfig()
audio_cfg = AudioConfig()
model_cfg = ModelConfig()
train_cfg = TrainConfig()
