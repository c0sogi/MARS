import os
import torch


class Config:
    # ==========================================
    # File Paths and Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_44"
    SUBMISSION_DIR = "./submission"

    # Input Data Paths (using generated Parquet metadata)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Specifications
    # ==========================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68
    NUM_TARGETS = 5

    # Input Feature Dimensions
    # 4 Nucleotides (A, G, C, U)
    # 3 Structure types ( (, ), . )
    # 7 Loop types (S, M, I, B, H, E, X)
    INPUT_CHANNELS = 4 + 3 + 7  # Total: 14

    # ==========================================
    # Model Architecture (Stabilized MLP-Gated Decoupled BiGRU)
    # ==========================================
    # Convolutional Stem
    CONV_FILTERS = 256
    CONV_KERNEL_SIZE = 3

    # Backbone
    HIDDEN_DIM = 384
    NUM_LAYERS = 3  # Optimal depth from Idea 17
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0  # Gradient clipping
    PATIENCE = 10  # Early stopping patience

    # ==========================================
    # Scoring and Metrics
    # ==========================================
    # Indices corresponding to: [reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
    # Scored columns: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    SCORING_INDICES = [0, 1, 3]

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
