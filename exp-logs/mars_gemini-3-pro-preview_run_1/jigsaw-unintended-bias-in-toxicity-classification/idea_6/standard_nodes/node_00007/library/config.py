import os
import torch


class Config:
    """
    Configuration class for Toxicity Classification with Bias Mitigation.
    Centralizes hyperparameters, file paths, and model settings.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True for fast debugging on a subset
    DEBUG_SAMPLE_SIZE = 5000
    NUM_WORKERS = 4  # Optimized for the 12 vCPU environment
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # File Paths
    # ==========================================
    # Metadata paths (pre-split and processed)
    TRAIN_PATH = "./metadata/train.csv"
    VAL_PATH = "./metadata/val.csv"
    TEST_PATH = "./metadata/test.csv"

    # Sample submission
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Output directories and files
    WORKING_DIR = "./working/idea_6"
    OUTPUT_DIR = WORKING_DIR  # Alias

    # Checkpoints and Cache
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.bin")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LEN = 256  # Sufficient for 95th percentile of text length
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    TRAIN_BATCH_SIZE = 16  # Adjusted for A100 and DeBERTa-base memory usage
    VALID_BATCH_SIZE = 32
    EPOCHS = 2
    LEARNING_RATE = 1.5e-5
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    WARMUP_RATIO = 0.05
    PATIENCE = 3  # For early stopping

    # ==========================================
    # Loss & Bias Mitigation Strategy
    # ==========================================
    # Loss Component Weights
    LAMBDA_RANK = 0.5  # Weight for the Margin Ranking Loss
    LAMBDA_AUX = 0.2  # Weight for the Auxiliary Identity/Subtype Heads

    # Sample Weights for Bias Traps
    # Applied to: (Non-Toxic + Identity) and (Toxic + No Identity)
    WEIGHT_BIAS_TRAP = 5.0
    WEIGHT_NORMAL = 1.0

    # Ranking Margin
    RANKING_MARGIN = 0.5

    # ==========================================
    # Data Columns
    # ==========================================
    TEXT_COL = "comment_text"
    TARGET_COL = "target"
    BINARY_TARGET_COL = "binary_target"

    # Identity Attributes (used for Aux Head 1 and Evaluation)
    IDENTITY_COLUMNS = [
        "male",
        "female",
        "homosexual_gay_or_lesbian",
        "christian",
        "jewish",
        "muslim",
        "black",
        "white",
        "psychiatric_or_mental_illness",
    ]

    # Toxicity Subtypes (used for Aux Head 2)
    AUX_COLUMNS = [
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack",
        "sexual_explicit",
    ]

    # Specific subtype for the Identity Attack Head
    IDENTITY_ATTACK_COL = "identity_attack"
