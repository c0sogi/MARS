import os
import torch


class Config:
    """
    Global configuration for the GC-DARN (Global-Context Direct-Access Recurrent Network) experiment.
    """

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea
    WORK_DIR = "./working/idea_57"

    # Ensure the working directory exists
    os.makedirs(WORK_DIR, exist_ok=True)

    # Metadata Files (Pre-generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Input Files
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output/Cache Files
    # Cache file for processed tensors (safe cache invalidation key)
    CACHE_FILE = os.path.join(WORK_DIR, "train_data_gc_darn_v1.npz")
    # Model checkpoint path
    MODEL_PATH = os.path.join(WORK_DIR, "best_model.pth")
    # Final submission file
    SUBMISSION_PATH = os.path.join(WORK_DIR, "submission.csv")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    SEQ_LENGTH = 107
    SCORED_LENGTH = 68

    # All target columns available in training (used for feedback loop context)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # The subset of targets that are actually scored in the competition metric
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Vocabulary mappings
    BASES = ["A", "G", "C", "U"]
    STRUCTURES = [".", "(", ")"]
    LOOP_TYPES = ["S", "M", "I", "B", "H", "E", "X"]

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Latent dimension for the backbone output
    LATENT_DIM = 64
    # Dimension for the feedback embeddings
    FEEDBACK_DIM = 32
    # Hidden dimension for the aggregation RNN
    HIDDEN_DIM = 64
    # Dropout rate
    DROPOUT = 0.1
    # Dilation schedule for the backbone
    DILATIONS = [1, 2, 4, 8, 16, 32]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Batch size set to 16 to ensure sufficient gradient updates (approx 108/epoch)
    BATCH_SIZE = 16
    # Learning rate
    LEARNING_RATE = 1e-3
    # Number of epochs
    EPOCHS = 25
    # Number of data loader workers
    NUM_WORKERS = 2
    # Random seed for reproducibility
    SEED = 42

    # Compute device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
