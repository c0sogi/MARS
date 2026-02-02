import os
import torch


class Config:
    """
    Configuration class for the High-Capacity Hierarchical Synthesis strategy.
    Centralizes file paths, model hyperparameters, and training settings.
    """

    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this strategy
    WORKING_DIR = "./working/idea_90"

    # Ensure working directory exists immediately
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata File Paths (Parquet format)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache File Paths (for deterministic data processing)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data.npz")

    # Output Paths
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Model Architecture
    # ==========================================
    # Input features: 4 (Nucleotide) + 3 (Structure) + 7 (Loop Type)
    INPUT_DIM = 14

    # Backbone Capacity (High-Capacity BiGRU)
    # Hidden dimension of 384 per direction = 768 Total
    HIDDEN_DIM = 768
    NUM_LAYERS = 4

    # Deep Residual Convolutional Stem
    # Kernel sizes for the stem layers: Conv(3) -> Res(5) -> Res(3)
    STEM_KERNEL_SIZES = [3, 5, 3]

    # Regularization
    DROPOUT = 0.1

    # ==========================================
    # Data Specifications
    # ==========================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Target Columns (All 5 ground truth conditions)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Columns used for official scoring (Validation Metric)
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32  # Tuned for A100 40GB with 768 dim model
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50

    # Optimization Stability
    # Mandatory gradient clipping to stabilize the massive hybrid architecture
    CLIP_GRAD_NORM = 1.0
    WEIGHT_DECAY = 1e-4

    # Scheduler settings (Cosine Annealing)
    T_MAX = NUM_EPOCHS

    # System settings
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Debug / Development
    # ==========================================
    # Flag to enable quick runs on subsets for debugging pipeline
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100
