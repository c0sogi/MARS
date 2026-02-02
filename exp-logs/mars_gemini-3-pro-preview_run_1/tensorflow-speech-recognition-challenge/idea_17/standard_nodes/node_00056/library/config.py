import os
import torch
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class AudioConfig:
    """
    Configuration for Audio Preprocessing (Log-Mel Spectrograms).
    """

    sample_rate: int = 16000
    duration: float = 1.0  # seconds
    n_fft: int = 1024  # 64ms window
    hop_length: int = 160  # 10ms stride
    n_mels: int = 128
    f_min: int = 0
    f_max: int = 8000  # Nyquist frequency

    # Computed property for input size
    @property
    def time_steps(self) -> int:
        # (sample_rate * duration) // hop_length + 1
        # 16000 // 160 + 1 = 101
        return int((self.sample_rate * self.duration) // self.hop_length + 1)


@dataclass
class TrainConfig:
    """
    Configuration for Training and Optimization.
    """

    # Hardware
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Batching
    batch_size: int = 128

    # Optimization
    lr: float = 1e-3
    min_lr: float = 5e-5  # 0.05 * lr
    weight_decay: float = 1e-2

    # SWA Strategy
    epochs: int = 25
    swa_start_epoch: int = 15
    swa_lr: float = 5e-5  # Constant LR for SWA phase (matches min_lr)

    # Regularization
    mixup_alpha: float = 1.0
    dropout_rate: float = 0.5
    multi_sample_dropout_count: int = 8

    # Checkpointing
    checkpoint_dir: str = "./working/idea_17"
    best_model_path: str = os.path.join(checkpoint_dir, "best_model.pth")
    swa_model_path: str = os.path.join(checkpoint_dir, "swa_model.pth")


@dataclass
class LabelConfig:
    """
    Configuration for Label Mapping and Taxonomy.
    Defines the 31+ Fine-Grained Classes.
    """

    # The 10 Target Commands
    target_labels: List[str] = field(
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
        ]
    )

    # The Silence Class
    silence_label: str = "silence"

    # The Auxiliary Commands (Standard Google Speech Commands v1/v2 set)
    # These are treated as distinct classes during training but mapped to 'unknown' for submission.
    aux_labels: List[str] = field(
        default_factory=lambda: [
            "bed",
            "bird",
            "cat",
            "dog",
            "eight",
            "five",
            "four",
            "happy",
            "house",
            "marvin",
            "nine",
            "one",
            "seven",
            "sheila",
            "six",
            "three",
            "tree",
            "two",
            "wow",
            "zero",
        ]
    )

    # Placeholder for unknown (used in submission mapping, not as a training class usually)
    unknown_label: str = "unknown"

    @property
    def all_classes(self) -> List[str]:
        """Returns the sorted list of all fine-grained classes used for training."""
        # Combine all lists
        classes = self.target_labels + [self.silence_label] + self.aux_labels
        # Sort to ensure deterministic index mapping
        return sorted(list(set(classes)))

    @property
    def num_classes(self) -> int:
        return len(self.all_classes)

    @property
    def label2id(self) -> Dict[str, int]:
        return {label: i for i, label in enumerate(self.all_classes)}

    @property
    def id2label(self) -> Dict[int, str]:
        return {i: label for label, i in self.label2id.items()}

    def map_to_submission_label(self, fine_grained_label: str) -> str:
        """
        Maps a fine-grained training label to the competition submission label.
        - Targets -> Targets
        - Silence -> Silence
        - Aux -> Unknown
        """
        if fine_grained_label in self.target_labels:
            return fine_grained_label
        if fine_grained_label == self.silence_label:
            return self.silence_label
        return self.unknown_label


@dataclass
class PathConfig:
    """
    File paths for data and metadata.
    """

    input_root: str = "./input"
    train_metadata: str = "./metadata/train.csv"
    val_metadata: str = "./metadata/val.csv"
    test_metadata: str = "./metadata/test.csv"
    submission_path: str = "./submission/submission.csv"
    cache_dir: str = "./working/idea_17"


# Instantiate global config objects
AUDIO_CONFIG = AudioConfig()
TRAIN_CONFIG = TrainConfig()
LABEL_CONFIG = LabelConfig()
PATH_CONFIG = PathConfig()

# Ensure cache directory exists
os.makedirs(PATH_CONFIG.cache_dir, exist_ok=True)
