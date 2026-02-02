import os
import torch


class Config:
    """
    Configuration class for the QA extraction task using XLM-R Large.
    Encapsulates model hyperparameters, training settings, data paths,
    and ensemble strategies.
    """

    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_NAME = "xlm-roberta-large"

    # =========================================================================
    # Data Processing & Tokenization
    # =========================================================================
    MAX_LENGTH = 384  # Maximum sequence length for the model
    DOC_STRIDE = 128  # Overlap between sliding windows

    # Negative Sampling Strategy
    # We use a 2:1 Negative-to-Positive ratio.
    # This controls how many negative (non-answer) windows are kept per positive window.
    NEGATIVE_SAMPLING_RATIO = 2.0

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Differential Learning Rates
    LR_BACKBONE = 1e-5  # Lower rate for the pre-trained transformer
    LR_HEAD = 5e-5  # Higher rate for the task-specific heads

    BATCH_SIZE = 4  # Small batch size for regularization/memory
    EPOCHS = 8  # Extended training duration for convergence

    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    max_grad_norm = 1.0

    # =========================================================================
    # Adversarial Training (FGM)
    # =========================================================================
    USE_FGM = True
    FGM_EPSILON = 1.0  # Perturbation magnitude

    # =========================================================================
    # Ensembling & Reproducibility
    # =========================================================================
    N_SEEDS = 5
    BASE_SEED = 42
    # Generate a list of seeds: [42, 43, 44, 45, 46]
    SEEDS = [BASE_SEED + i for i in range(N_SEEDS)]

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    def __init__(self):
        """
        Initialize configuration and ensure necessary directories exist.
        """
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

        # Print config summary for verification
        print(f"Config initialized: {self.MODEL_NAME}")
        print(f"Working Directory: {self.WORKING_DIR}")
        print(f"Device: {self.DEVICE}")
