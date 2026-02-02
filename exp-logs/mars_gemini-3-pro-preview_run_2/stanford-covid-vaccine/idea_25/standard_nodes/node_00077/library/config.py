import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # General Configuration
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    # Debugging
    DEBUG = False
    DEBUG_SAMPLES = 100

    # --------------------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------------------
    # Input Metadata (Read-Only)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory (Write Access)
    # All outputs and cache files must go here
    WORKING_DIR = "./working/idea_25"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Cache Files
    # Using specific version suffix 'stacked_interaction_v1' to enforce
    # re-generation of features (specifically Partner Identity)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_stacked_interaction_v1.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data_stacked_interaction_v1.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data_stacked_interaction_v1.npz")

    # Model Artifacts
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Specifications
    # --------------------------------------------------------------------------
    SEQ_LENGTH = 107
    SCORING_LENGTH = 68

    # Input Feature Calculation:
    # 1. Sequence (One-Hot): 4
    # 2. Structure (One-Hot): 3
    # 3. Loop Type (One-Hot): 7
    # 4. Partner Identity (One-Hot): 5 (A, G, C, U, None)
    # Total Input Channels = 19
    INPUT_DIM = 19

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    # Backbone: Dense Dilated TCN
    BACKBONE_GROWTH_RATE = 64
    BACKBONE_DILATIONS = [1, 2, 4, 8, 16, 32]
    BACKBONE_KERNEL_SIZE = 3
    DROPOUT = 0.1

    # Latent Interaction Module
    LATENT_DIM = 32  # 1x1 Conv projection dimension

    # Post-Interaction Refinement (Mini-DenseNet)
    # Input to this block is Concatenation(Local, Partner) = 64 channels
    REFINEMENT_GROWTH_RATE = 32
    # List of (kernel_size, dilation) tuples
    REFINEMENT_LAYERS = [(3, 1), (3, 2)]

    # Global Aggregation (BiGRU)
    # Input dim = 64 (Fused) + 32 (Refine L1) + 32 (Refine L2) = 128
    RNN_HIDDEN_DIM = 64  # Bidirectional, so output will be 128
    RNN_LAYERS = 1

    # Output Head
    NUM_TARGETS = 5

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 16
    EPOCHS = 25
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5  # For early stopping or LR reduction

    # Metric Calculation
    # Indices corresponding to: reactivity, deg_Mg_pH10, deg_Mg_50C
    # (deg_pH10 is index 2, deg_50C is index 4 - these are auxiliary)
    SCORED_TARGET_INDICES = [0, 1, 3]
