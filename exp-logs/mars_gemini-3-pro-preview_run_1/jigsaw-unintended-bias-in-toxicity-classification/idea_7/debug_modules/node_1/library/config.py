import os
import torch


class Config:
    # ====================================================
    # General Settings
    # ====================================================
    SEED = 42
    DEBUG = False  # Set to True for fast debugging runs (subsamples data)
    EXP_NAME = "idea_7"
    OUTPUT_DIR = f"./working/{EXP_NAME}/"

    # Create output directory immediately
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ====================================================
    # Data Paths
    # ====================================================
    # Using metadata paths as defined in the provided documentation
    TRAIN_PATH = "./metadata/train.csv"
    VAL_PATH = "./metadata/val.csv"
    TEST_PATH = "./metadata/test.csv"
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # ====================================================
    # Model Architecture
    # ====================================================
    MODEL_NAME = "microsoft/deberta-v3-large"
    MAX_LEN = (
        320  # Sequence length; 320 covers most comments without excessive truncation
    )
    DROPOUT = 0.1
    NUM_CLASSES = 1  # Primary target: Toxicity

    # Multi-Task Learning Heads
    # 1. Identity Attributes (Auxiliary Head)
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
    # 2. Identity Attack Subtype (Auxiliary Head)
    AUX_COLS = ["identity_attack"]

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    EPOCHS = 3
    # A100 40GB allows for decent batch sizes with Large models.
    # Effective batch size = TRAIN_BATCH_SIZE * ACCUMULATION_STEPS
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 16
    ACCUMULATION_STEPS = 2

    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    SCHEDULER_TYPE = "cosine"

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True
    FP16 = True  # Use Mixed Precision

    # ====================================================
    # Hybrid Loss & Sampling Strategy
    # ====================================================
    # Loss Coefficients
    ALPHA_RANK = 0.5  # Weight for Pairwise Margin Ranking Loss
    BETA_AUX = 0.2  # Weight for Auxiliary Heads (Identities + Attack)

    # Ranking Loss Settings
    RANKING_MARGIN = 0.5

    # Weighted Sampling / Loss Weighting
    # "Bias Traps" are examples where the model is likely to fail (e.g., Non-Toxic + Identity Mention).
    # We apply a higher weight to these samples to force the model to learn them.
    BIAS_TRAP_WEIGHT = 5.0
    NORMAL_WEIGHT = 1.0

    # ====================================================
    # Adversarial Weight Perturbation (AWP)
    # ====================================================
    USE_AWP = True
    AWP_START_EPOCH = (
        1.0  # Start AWP after the first epoch to allow initial convergence
    )
    AWP_LR = 1e-4  # Learning rate for the adversarial perturbation (ascent)
    AWP_EPS = 1e-2  # Epsilon: maximum magnitude of weight perturbation
