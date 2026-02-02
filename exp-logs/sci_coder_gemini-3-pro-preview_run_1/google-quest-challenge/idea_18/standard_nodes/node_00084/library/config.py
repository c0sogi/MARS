import os
import torch


class Config:
    """
    Configuration class for the Hybrid-Depth Asymmetric Dual-Encoder model.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_18"

    # Input files (using metadata splits)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output files
    SUBMISSION_PATH = "./submission/submission.csv"
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Asymmetric Backbone Architecture
    QUESTION_MODEL_NAME = (
        "roberta-base"  # 12 layers, for complex hierarchical structure
    )
    ANSWER_MODEL_NAME = "distilroberta-base"  # 6 layers, for simpler structure
    TOKENIZER_NAME = "roberta-base"  # Shared tokenizer

    HIDDEN_SIZE = 768  # Shared hidden dimension
    NUM_TARGETS = 30  # Number of target labels

    # =========================================================================
    # Data Processing
    # =========================================================================
    MAX_LEN_Q = 512  # Max sequence length for Question branch
    MAX_LEN_A = 512  # Max sequence length for Answer branch

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Batch Size Strategy
    TRAIN_BATCH_SIZE = 8  # Physical batch size (fits in GPU memory)
    VALID_BATCH_SIZE = 16  # Validation batch size
    ACCUMULATION_STEPS = 2  # Effective batch size = 8 * 2 = 16

    # Phantom Scheduling Strategy
    # Schedule decays as if training for 7 epochs, but we stop at 3.
    # This preserves the optimal decay curve for fine-tuning.
    EPOCHS_SCHEDULE = 7
    EPOCHS_ACTUAL = 3

    # Optimization
    # Differential Learning Rates
    LR_HEAD = 1e-3  # High LR for the initialized head
    LR_BACKBONE = 2e-5  # Low LR for the pre-trained backbones

    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # Warmup Strategy
    # Freeze backbones for the first epoch to align the head
    WARMUP_EPOCHS = 1
