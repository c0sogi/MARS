import os
import torch


class Config:
    """
    Configuration class for the RNA degradation prediction task.
    Centralizes hyperparameters, file paths, and execution settings.
    """

    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_45"

    # Ensure the working directory exists immediately
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Files (Parquet)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")

    # Raw Input Files
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache Files (Processed Tensors)
    # Using .npz for efficient storage of numpy arrays before conversion to Tensor
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_cache.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_cache.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_cache.npz")

    # Output Artifacts
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission.csv"  # Final submission location

    # ==========================================
    # Data Specifications
    # ==========================================
    SEQ_LEN = 107
    PRED_LEN = 68
    NUM_TARGETS = 5

    # Feature Dimensions (One-Hot Encodings)
    # Nucleotides: A, G, C, U -> 4
    # Structure: (, ), . -> 3
    # Loop Type: S, M, I, B, H, E, X -> 7
    INPUT_DIM = 4 + 3 + 7  # Total: 14

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Deep Stabilized Decoupled BiGRU (DSD-BiGRU)
    HIDDEN_DIM = 384  # High capacity within safe limits
    NUM_LAYERS = 4  # Deep architecture
    DROPOUT = 0.1

    # Convolutional Stem
    CONV_FILTERS = 256
    CONV_KERNEL_SIZE = 3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32  # Safe batch size for 220GB RAM / A100
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # Standard regularization
    GRAD_CLIP = 1.0  # Mandatory for deep RNN stability

    # Optimization
    PATIENCE = 7  # Early stopping patience
    NUM_WORKERS = 4  # Data loading workers

    # Device Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
