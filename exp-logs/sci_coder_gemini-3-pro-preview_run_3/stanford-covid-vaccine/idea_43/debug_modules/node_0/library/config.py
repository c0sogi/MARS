import os
import torch


class Config:
    """
    Configuration for the Deep Bias-Refined Decoupled Post-Norm BiGRU strategy.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_43"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Create working directory immediately upon import to ensure availability
    os.makedirs(WORKING_DIR, exist_ok=True)
    # Ensure submission directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Input Channels:
    # 4 (Sequence: A, G, C, U)
    # + 3 (Structure: (, ), .)
    # + 7 (Loop Type: S, M, I, B, H, E, X)
    IN_CHANNELS = 14

    # All Target Columns available in training data
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Columns specifically used for the MCRMSE scoring metric
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Backbone: 4-Layer BiGRU with Decoupled Post-Norm Structural Injection
    NUM_LAYERS = 4
    HIDDEN_DIM = 384

    # Convolutional Stem
    STEM_FILTERS = 256
    STEM_KERNEL_SIZE = 3

    # Regularization
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Gradient Clipping (Critical for stability in deep hybrid architectures)
    CLIP_GRAD_NORM = 1.0

    # Early Stopping
    PATIENCE = 10

    # Scheduler settings (Cosine Annealing)
    MIN_LR = 1e-6

    # =========================================================================
    # System & Reproducibility
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # =========================================================================
    # Debugging & Development
    # =========================================================================
    # Set to True to train on a small subset for rapid testing
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100
