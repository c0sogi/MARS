import os
import torch


class Config:
    """
    Configuration for the Anchored Hybrid-Input Recurrent Network (AHI-RN).
    Defines paths, hyperparameters, and constants.
    """

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Idea-Specific Directory (Cache & Model Checkpoints)
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_60")
    os.makedirs(IDEA_DIR, exist_ok=True)

    # Metadata Inputs
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache Files (NPZ format)
    # Using specific version tag 'v1' to ensure cache invalidation if needed
    TRAIN_CACHE_PATH = os.path.join(IDEA_DIR, "train_data_ahi_rn_v1.npz")
    VAL_CACHE_PATH = os.path.join(IDEA_DIR, "val_data_ahi_rn_v1.npz")
    TEST_CACHE_PATH = os.path.join(IDEA_DIR, "test_data_ahi_rn_v1.npz")

    # Model Output
    MODEL_SAVE_PATH = os.path.join(IDEA_DIR, "best_model.pth")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # All ground truth columns available in training data
    ALL_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # The subset of targets that are actually scored in the competition
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    NUM_TARGETS = len(ALL_TARGETS)

    # =========================================================================
    # Model Architecture Hyperparameters
    # =========================================================================
    # Backbone
    HIDDEN_DIM = 64  # Constraint: Compact Hidden Size
    LATENT_DIM = 64  # Projection dimension before RNN
    KERNEL_SIZE = 3
    DROPOUT = 0.1

    # Dilated TCN Configuration
    # Exponentially increasing dilation rates
    DILATIONS = [1, 2, 4, 8, 16, 32]
    NUM_LAYERS = len(DILATIONS)

    # Feedback Module
    FEEDBACK_EMBED_DIM = 32  # Dimension of feedback embeddings

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 16  # Constraint: Maximize gradient update frequency
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 25
    PATIENCE = 5  # Early stopping patience

    # Loss Weights
    # L_total = MCRMSE(Y_final) + AUX_WEIGHT * MCRMSE(Y_aux)
    AUX_LOSS_WEIGHT = 0.5

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2
