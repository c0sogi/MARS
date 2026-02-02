import os
import torch


class Config:
    # ==============================
    # Paths & Directories
    # ==============================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_52"
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.csv")
    VAL_FILE = os.path.join(METADATA_DIR, "val.csv")
    TEST_FILE = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache Files (Explicit Cache Invalidation for SS-PFN Idea)
    # Using 'ss_pfn_v1' tag to ensure new features (Partner Identity + Spatial Stem readiness) are generated.
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_ss_pfn_v1.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data_ss_pfn_v1.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data_ss_pfn_v1.npz")

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==============================
    # Data Parameters
    # ==============================
    SEQ_LENGTH = 107
    SCORED_SEQ_LENGTH = 68

    # The columns actually used for the competition metric
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # All target columns present in the data (needed for parsing)
    ALL_TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # ==============================
    # Model Hyperparameters
    # ==============================
    # Input Stem
    STEM_KERNEL_SIZE = 3  # Spatial input stem

    # Backbone (Dense Dilated TCN)
    HIDDEN_DIM = 64
    GROWTH_RATE = 64
    DILATIONS = [1, 2, 4, 8, 16, 32]  # 6 Layers
    DROPOUT = 0.1
    LATENT_DIM = 64

    # Pure-Feedback Module
    FEEDBACK_GROWTH_RATE = 16  # Low capacity for feedback loop

    # Interaction & Aggregation
    RNN_HIDDEN_DIM = 64  # Compact hidden size

    # ==============================
    # Training Hyperparameters
    # ==============================
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    NUM_WORKERS = 2
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Iterative Refinement Loss Weight
    # Total Loss = Loss(Pass2) + PASS1_WEIGHT * Loss(Pass1)
    PASS1_WEIGHT = 0.5

    @classmethod
    def setup(cls):
        """Ensures working and submission directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
