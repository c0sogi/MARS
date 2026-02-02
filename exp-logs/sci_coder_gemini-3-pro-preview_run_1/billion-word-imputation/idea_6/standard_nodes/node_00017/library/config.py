import os
import torch


class Config:
    """
    Configuration for the Bifurcated Interleaved Transformer (Idea 6).
    """

    # --------------------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42

    # --------------------------------------------------------------------------
    # Paths & Directories
    # --------------------------------------------------------------------------
    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for this specific idea/iteration
    WORKING_DIR = "./working/idea_6"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Model and Vocabulary Artifacts
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.npy")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data & Vocabulary Settings
    # --------------------------------------------------------------------------
    VOCAB_SIZE = 50000
    MIN_FREQ = 2
    MAX_SEQ_LEN = 128  # Maximum sequence length (including GAPs)

    # Special Tokens
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    GAP_TOKEN = "[GAP]"

    # --------------------------------------------------------------------------
    # Model Architecture: Bifurcated Interleaved Transformer
    # --------------------------------------------------------------------------
    EMBED_DIM = 256
    NHEAD = 8
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1

    # Split-Stream Architecture Configuration
    SHARED_LAYERS = 6  # Bottom K layers (Context Encoder)
    BRANCH_LAYERS = 2  # Top layers for Localization and Identification streams

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 128  # Optimized for A100 40GB
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    PATIENCE = 3  # Early stopping patience

    # Decoupled Multi-Task Loss Weights
    LAMBDA_LOC = 1.0  # Weight for Binary Localization Loss (Gap Detection)
    LAMBDA_ID = 1.0  # Weight for Masked Cross-Entropy Loss (Word Identification)

    # --------------------------------------------------------------------------
    # Compute & Hardware
    # --------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers
