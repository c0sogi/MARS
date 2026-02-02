import os
import torch


class Config:
    # ==========================================
    # 1. Paths and Directories
    # ==========================================
    # Input Data (Metadata Parquet Files)
    TRAIN_DATA_PATH = "./metadata/train.parquet"
    VAL_DATA_PATH = "./metadata/val.parquet"
    TEST_DATA_PATH = "./metadata/test.parquet"
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Working Directory (Idea 92)
    WORKING_DIR = "./working/idea_92"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Cache Paths
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_cache.npy")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_cache.npy")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_cache.npy")

    # ==========================================
    # 2. Data Specifications
    # ==========================================
    SEED = 42
    SEQ_LEN = 107
    SEQ_SCORED = 68

    # Mappings for One-Hot Encoding
    # Nucleotides (4 channels)
    TOKEN2INT = {x: i for i, x in enumerate("AGUC")}
    # Structure (3 channels)
    STRUCT2INT = {x: i for i, x in enumerate("().")}
    # Predicted Loop Type (7 channels)
    LOOP2INT = {x: i for i, x in enumerate("SMIBHEX")}

    # Total Input Channels = 4 + 3 + 7 = 14
    INPUT_DIM = 14

    # Target Columns
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Columns used for Validation Score (MCRMSE)
    # Note: Only these 3 are scored in the competition metric
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # ==========================================
    # 3. Model Architecture (High-Capacity)
    # ==========================================
    # Dilated Residual Stem
    STEM_KERNEL_SIZE = 3
    STEM_DILATIONS = [1, 2, 4]  # Hierarchical context aggregation
    STEM_FILTERS = 768  # Project to high dimension immediately

    # BiGRU Backbone
    RNN_HIDDEN_DIM = 384  # Per direction (Total = 768)
    RNN_LAYERS = 4  # Deep hierarchy

    # Interaction Module (GLU-Decoupled)
    USE_GLU_INTERACTION = True

    # Regularization
    DROPOUT = 0.1  # Conservative dropout

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32  # Adjust based on A100 memory (40GB is plenty for 32)
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # Standard for AdamW

    # Optimization Stability
    MAX_GRAD_NORM = 1.0  # Mandatory for deep hybrid architectures

    # Scheduler
    T_MAX = NUM_EPOCHS  # For CosineAnnealingLR

    # Debugging / Development
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SUBSET_SIZE = 100  # Number of samples to use in debug mode

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For DataLoader
