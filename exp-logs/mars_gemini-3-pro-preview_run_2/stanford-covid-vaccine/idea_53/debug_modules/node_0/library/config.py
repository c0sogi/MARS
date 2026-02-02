import os
import torch


class Config:
    # =========================================================================
    # Directories and File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_53"

    # Raw Input Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Stratified Splits)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Cache Files (Explicit Cache Invalidation with new version suffix)
    # Using .npz for numpy array storage as requested
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_ss_dfrn_v1.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data_ss_dfrn_v1.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data_ss_dfrn_v1.npz")

    # Output Paths
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Dimensions and Columns
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Target Columns in the dataset
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Columns used for the competition metric and loss calculation
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # =========================================================================
    # Feature Engineering Parameters
    # =========================================================================
    # Input Features:
    # 1. Sequence One-Hot (A, G, C, U) -> 4
    # 2. Structure One-Hot ((, ), .) -> 3
    # 3. Predicted Loop Type One-Hot (S, M, I, B, H, E, X) -> 7
    # 4. Partner Identity One-Hot (A, G, C, U) -> 4 (Explicit injection)
    # Total Input Channels = 4 + 3 + 7 + 4 = 18
    NUM_NODE_FEATURES = 18

    # =========================================================================
    # Model Hyperparameters (SS-DFRN)
    # =========================================================================
    # 1. Spatial Input Stem
    STEM_KERNEL_SIZE = 3

    # 2. Main Backbone (Dense Dilated TCN)
    # Post-Activation Micro-Architecture
    BACKBONE_DILATIONS = [1, 2, 4, 8, 16, 32]
    BACKBONE_GROWTH_RATE = 64
    BACKBONE_KERNEL_SIZE = 3
    LATENT_DIM = 64  # Z dimension
    DROPOUT = 0.1

    # 3. Lightweight Dense Feedback Module
    FEEDBACK_INPUT_CHANNELS = 5  # Recycled predictions (y_hat)
    FEEDBACK_GROWTH_RATE = 16  # Low capacity constraint
    FEEDBACK_OUTPUT_CHANNELS = 32  # E_fb dimension

    # 4. Interaction & Aggregation
    RNN_HIDDEN_SIZE = 64  # Compact size
    RNN_BIDIRECTIONAL = True
    RNN_LAYERS = 1

    # =========================================================================
    # Training Configuration
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization
    BATCH_SIZE = 16  # Fits comfortably in A100
    EPOCHS = 50  # Sufficient with early stopping
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10  # Early stopping patience

    # Loss Weights
    # L_total = MCRMSE(Y2) + 0.5 * MCRMSE(Y1)
    LOSS_PASS2_WEIGHT = 1.0
    LOSS_PASS1_WEIGHT = 0.5

    # Debugging / Development
    # Set to None to use full dataset, or an integer to limit rows
    DEBUG_SUBSET_SIZE = None

    @classmethod
    def setup(cls):
        """Ensures the working directory exists."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)


# Initialize environment
Config.setup()
