import os
import torch


class Config:
    """
    Configuration class for the RNA Degradation Prediction task.
    Implements settings for the Masked-Reconstruction Regularized BiGRU strategy.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Input Data Paths (Parquet Metadata)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Submission Template
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Data Specifications
    # ==========================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Feature Vocabularies (One-Hot Encoding Sizes)
    # Sequence: A, G, C, U
    VOCAB_SIZE_SEQ = 4
    # Structure: ., (, )
    VOCAB_SIZE_STRUCT = 3
    # Predicted Loop Type: S, M, I, B, H, E, X
    VOCAB_SIZE_LOOP = 7

    # Total Input Channels
    INPUT_DIM = VOCAB_SIZE_SEQ + VOCAB_SIZE_STRUCT + VOCAB_SIZE_LOOP  # 14

    # Targets
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    SCORED_COLS_INDICES = [0, 1, 3]
    OUTPUT_DIM = len(TARGET_COLS)  # 5

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Architecture: CNN Stem -> BiGRU Backbone -> Dual Heads
    CNN_FILTERS = 256
    CNN_KERNEL_SIZE = 3

    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BIDIRECTIONAL = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 64
    EPOCHS = 50

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler (CosineAnnealingLR)
    SCHEDULER_T_MAX = EPOCHS
    SCHEDULER_MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # ==========================================
    # Strategy Specifics (Masked Reconstruction)
    # ==========================================
    # Probability of masking an input position (vector becomes zero) during training
    MASK_PROB = 0.15

    # Weight (Lambda) for the auxiliary CrossEntropy reconstruction loss
    # Total Loss = MCRMSE + (RECON_LOSS_WEIGHT * CrossEntropy)
    RECON_LOSS_WEIGHT = 0.5

    # ==========================================
    # Hardware & System
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """Ensures necessary working directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)


# Initialize directories upon import
Config.setup()
