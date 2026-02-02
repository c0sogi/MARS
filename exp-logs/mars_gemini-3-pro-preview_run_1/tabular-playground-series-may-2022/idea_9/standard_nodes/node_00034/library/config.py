import os
import torch
from dataclasses import dataclass


@dataclass
class Config:
    """
    Configuration for the Discriminative Granular Unified Transformer (DiGUT) pipeline.
    """

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    INPUT_DIR: str = "./input"
    METADATA_DIR: str = "./metadata"
    WORKING_DIR: str = "./working/idea_9"
    SUBMISSION_DIR: str = "./submission"

    # Data Paths (using metadata splits)
    TRAIN_PATH: str = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH: str = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH: str = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH: str = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_PATH: str = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH: str = os.path.join(WORKING_DIR, "digut_model.pth")

    # Cache Paths
    CACHE_DIR: str = os.path.join(WORKING_DIR, "cache")

    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    SEED: int = 42
    NUM_WORKERS: int = 12  # Utilizing available vCPUs
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # Model Architecture (DiGUT)
    # -------------------------------------------------------------------------
    HIDDEN_DIM: int = 256
    NUM_LAYERS: int = 4
    NUM_HEADS: int = 8
    FORWARD_DIM: int = 1024  # Typically 4 * HIDDEN_DIM
    DROPOUT: float = 0.1
    ATTENTION_DROPOUT: float = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Optimized for A100-40GB. Large batch size for stability and speed.
    # Adjusted for 16GB GPU environment (Cite debug_lesson_3)
    BATCH_SIZE: int = 1024
    EPOCHS: int = 30

    # Optimizer (AdamW)
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 0.01

    # Scheduler (OneCycleLR)
    PCT_START: float = 0.1  # Warmup percentage

    # -------------------------------------------------------------------------
    # Auxiliary Task & Regularization
    # -------------------------------------------------------------------------
    # Discriminative Detection Task
    AUX_WEIGHT: float = 0.5  # Lambda: Weight for the discriminator loss
    SWAP_PROB: float = 0.15  # Probability of swapping a token (Swap Noise)

    # Target Loss
    LABEL_SMOOTHING: float = 0.01

    def __post_init__(self):
        """
        Ensure necessary directories exist upon initialization.
        """
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    def display(self):
        """
        Prints the configuration.
        """
        print("=" * 30)
        print("CONFIG")
        print("=" * 30)
        for field in self.__dataclass_fields__:
            print(f"{field}: {getattr(self, field)}")
        print("=" * 30)
