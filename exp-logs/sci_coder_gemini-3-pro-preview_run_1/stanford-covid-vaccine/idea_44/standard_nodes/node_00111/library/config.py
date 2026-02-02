import os
import torch


class Config:
    """
    Configuration class for the RNA Degradation Prediction task.
    Implements the 'Dual-Stream Recurrent Fusion Wide-Stream BiGRU' strategy.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory for the current idea (idea_44)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_44")

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # File Paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Targets to be trained on (excluding deg_pH10 and deg_50C as per instructions)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # =========================================================================
    # Model Architecture Hyperparameters
    # =========================================================================
    # Embeddings
    EMB_DIM_SEQ = 128  # Atomic Sequence
    EMB_DIM_LOOP = 64  # Predicted Loop Type
    EMB_DIM_DIST = 64  # Signed Sinusoidal Pairing Distance

    # Dual-Stream Recurrent Stem
    # Two parallel BiGRUs: Sequence Stem & Structure Stem
    # Each projects to a hidden state of size 192.
    # For BiGRU, hidden_size per direction = output_dim / 2
    STEM_OUTPUT_DIM = 192
    STEM_HIDDEN_SIZE = STEM_OUTPUT_DIM // 2

    # Backbone: Wide-Stream Residual Blocks
    # Fused width = 192 (Seq) + 192 (Struct) = 384
    BACKBONE_HIDDEN_SIZE = 384
    BACKBONE_LAYERS = 6
    DROPOUT = 0.2  # Inter-layer dropout (applied after BiGRU, before residual)

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    CLIP_GRAD = 1.0

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    @staticmethod
    def create_dirs():
        """Ensures necessary directories exist."""
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
