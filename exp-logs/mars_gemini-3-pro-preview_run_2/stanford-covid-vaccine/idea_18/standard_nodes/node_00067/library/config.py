import os
import torch


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_18"

    # Ensure the working directory exists for caching
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Unique cache version to enforce regeneration of preprocessed data
    # This ensures 'Partner Identity' and 'Partner Index' features are created.
    CACHE_VERSION = "interaction_enriched_v1"

    SEQ_LENGTH = 107
    PRED_LEN = 68

    # The 5 ground truth columns provided in training data
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Indices of the targets that are scored by the competition metric (MCRMSE)
    # 0: reactivity, 1: deg_Mg_pH10, 3: deg_Mg_50C
    SCORED_TARGET_INDICES = [0, 1, 3]

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Input Feature Dimensions:
    # Sequence (4) + Structure (3) + Loop Type (7) + Partner Identity (4) = 18
    INPUT_DIM = 18

    # Dense Dilated TCN Backbone
    GROWTH_RATE = 64  # Strictly set to 64 to manage capacity
    KERNEL_SIZE = 3
    DILATIONS = [1, 2, 4, 8, 16, 32]  # Exponential dilation for global context
    DROPOUT = 0.1

    # Interaction Enrichment Module
    # Dimension to compress features to before the gather/interaction step
    LATENT_DIM = 32

    # BiGRU Head
    # The enrichment module produces a concatenated vector of size 4 * LATENT_DIM = 128
    # (Local + Partner + Product + Difference)
    # We set hidden size to half of this to maintain dimension in the bidirectional output
    GRU_HIDDEN_DIM = 64
    GRU_LAYERS = 1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    EPOCHS = 25
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5  # Early stopping patience

    # =========================================================================
    # System & Reproducibility
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2
    SEED = 42
