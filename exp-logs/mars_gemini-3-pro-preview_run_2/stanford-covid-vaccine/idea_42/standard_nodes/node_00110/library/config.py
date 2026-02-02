import os
import torch


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_42"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Stratified Splits)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Cache Files (Preprocessed Data)
    # Using specific versioning for DR-RHN to ensure cache safety
    # Cite Lesson 32: Explicit versioning to prevent stale data usage
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_dr_rhn_v2.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data_dr_rhn_v2.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data_dr_rhn_v2.npz")

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Dimensions & Structure
    # =========================================================================
    SEQ_LENGTH = 107
    SCORED_SEQ_LENGTH = 68

    # Input Features
    # 4 (Seq) + 3 (Struct) + 7 (Loop) + 4 (Partner Identity) = 18
    NUM_NODE_FEATURES = 18

    # Targets
    # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    NUM_TARGETS = 5
    # Only these are used for the metric: reactivity, deg_Mg_pH10, deg_Mg_50C
    SCORED_TARGET_INDICES = [0, 1, 3]

    # =========================================================================
    # Model Hyperparameters (DR-RHN)
    # =========================================================================
    # 1. Main Backbone (Static Dense Dilated TCN)
    BACKBONE_GROWTH_RATE = 64
    BACKBONE_KERNEL_SIZE = 3
    BACKBONE_LAYERS = 6  # Dilations: 1, 2, 4, 8, 16, 32
    BACKBONE_DROPOUT = 0.1
    LATENT_DIM = 64  # Z dimension

    # 2. Lightweight Dense Feedback Module
    FEEDBACK_IN_CHANNELS = 5  # Number of predicted targets
    FEEDBACK_GROWTH_RATE = 12  # Lightweight constraint
    FEEDBACK_LAYERS = 4  # Depth of feedback processing
    FEEDBACK_HIDDEN_DIM = 16  # Initial projection before dense block
    FEEDBACK_EMBED_DIM = 32  # E_fb dimension

    # 3. Aggregation (RNN)
    RNN_HIDDEN_DIM = 128
    RNN_LAYERS = 1
    RNN_BIDIRECTIONAL = True

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 16
    EPOCHS = 25
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5  # Early stopping patience

    # Loss Weights
    # L_total = MCRMSE(Y_final) + AUX_WEIGHT * MCRMSE(Y_intermediate)
    AUX_LOSS_WEIGHT = 0.5

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    # =========================================================================
    # Utility Functions
    # =========================================================================
    @staticmethod
    def get_device():
        return Config.DEVICE

    def __str__(self):
        return str(self.__class__.__dict__)
