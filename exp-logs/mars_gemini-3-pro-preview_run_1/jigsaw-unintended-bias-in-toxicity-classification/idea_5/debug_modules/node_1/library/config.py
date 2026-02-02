import os
import torch


class Config:
    # ==========================================
    # Experiment Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to limit dataset size for quick debugging

    # ==========================================
    # Data Paths
    # ==========================================
    # Using metadata files generated in the previous step
    TRAIN_META_PATH = "./metadata/train.csv"
    VAL_META_PATH = "./metadata/val.csv"
    TEST_META_PATH = "./metadata/test.csv"

    # Sample submission for format reference
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Output directories
    WORKING_DIR = "./working/idea_5"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_OUTPUT_DIR = os.path.join(WORKING_DIR, "model_checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LEN = 256  # Sufficient for 95th percentile of text length

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # A100 40GB can handle larger batches, but 32 is a safe baseline for DeBERTa-base
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 64
    EPOCHS = 2
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.06
    MAX_GRAD_NORM = 1.0

    # ==========================================
    # Semantic Triangulation & Bias Loss Config
    # ==========================================
    # Loss Weights for Multi-Task Learning
    # Total Loss = L_primary + (lambda1 * L_identity) + (lambda2 * L_attack)
    LAMBDA_IDENTITY = 0.5
    LAMBDA_ATTACK = 0.5

    # Bias-Centric Weighting
    # Multiplier for examples in "Bias Trap" subgroups (Toxic+Identity or NonToxic+Identity)
    BIAS_WEIGHT_MULTIPLIER = 5.0

    # ==========================================
    # Column Definitions
    # ==========================================
    TARGET_COL = "target"
    TEXT_COL = "comment_text"

    # Subtype used for the "Identity Attack" auxiliary head
    IDENTITY_ATTACK_COL = "identity_attack"

    # Identity columns for the "Identity" auxiliary head and evaluation
    IDENTITY_COLS = [
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

    # ==========================================
    # Hardware & Environment
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    @staticmethod
    def setup_directories():
        """Creates necessary directories for outputs and caching."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.MODEL_OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when config is imported
Config.setup_directories()
