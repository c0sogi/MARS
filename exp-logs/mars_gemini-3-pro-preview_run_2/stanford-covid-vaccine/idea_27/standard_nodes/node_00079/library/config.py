import os
import torch


class Config:
    # =========================================================================
    # Directories and File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORK_DIR = "./working/idea_27"

    # Ensure working directory exists
    os.makedirs(WORK_DIR, exist_ok=True)

    # Metadata Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache File Paths (Explicit unique names for this idea)
    TRAIN_CACHE = os.path.join(WORK_DIR, "train_data_projected_dense_v1.npz")
    VAL_CACHE = os.path.join(WORK_DIR, "val_data_projected_dense_v1.npz")
    TEST_CACHE = os.path.join(WORK_DIR, "test_data_projected_dense_v1.npz")

    # Model Checkpoint Path
    MODEL_PATH = os.path.join(WORK_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEED = 42
    SEQ_LEN = 107
    PRED_LEN = 68

    # Input Feature Dimensions
    # Sequence (4) + Structure (3) + LoopType (7) + PartnerIdentity (5)
    # PartnerIdentity includes A, G, C, U, and None (for unpaired)
    INPUT_DIM = 4 + 3 + 7 + 5

    # Target Columns
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Columns used for scoring (and loss calculation)
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = 5

    # =========================================================================
    # Model Hyperparameters (Projected Latent-Interaction Dense Network)
    # =========================================================================
    # DenseNet Backbone
    GROWTH_RATE = 64  # Width of conv layers in dense blocks
    KERNEL_SIZE = 3
    DILATIONS = [1, 2, 4, 8, 16, 32]  # Exponential dilation rates
    DROPOUT = 0.1

    # Interaction Layer
    LATENT_DIM = 128  # Dimension for projection before gathering

    # RNN Head
    RNN_HIDDEN_DIM = 128  # Bidirectional GRU hidden size (output will be 256)
    RNN_LAYERS = 1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50  # Max epochs
    PATIENCE = 10  # Early stopping patience
    WEIGHT_DECAY = 1e-4  # Regularization

    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # For DataLoader

    @staticmethod
    def print_config():
        """Prints the current configuration."""
        print("=" * 40)
        print(f"CONFIG: {Config.WORK_DIR}")
        print("=" * 40)
        print(f"Device: {Config.DEVICE}")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Learning Rate: {Config.LEARNING_RATE}")
        print(f"Growth Rate: {Config.GROWTH_RATE}")
        print(f"Latent Dim: {Config.LATENT_DIM}")
        print(f"Scored Targets: {Config.SCORED_COLS}")
        print("=" * 40)
