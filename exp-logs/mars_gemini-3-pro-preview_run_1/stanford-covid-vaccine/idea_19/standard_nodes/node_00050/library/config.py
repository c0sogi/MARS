import os
import torch


class Config:
    # =========================================================================
    # Directory and File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_19"
    SUBMISSION_DIR = "./submission"

    # Metadata Files (Parquet format as generated in previous steps)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Sample Submission
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Ensure necessary write directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LENGTH = 107
    PRED_LENGTH = 68

    # Targets: Strictly training on the 3 scored columns
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Vocabulary Sizes
    # Nucleotides: A, G, C, U
    VOCAB_SIZE_SEQ = 4
    # Predicted Loop Types: S, M, I, B, H, E, X
    VOCAB_SIZE_LOOP = 7

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Architecture: Scalar-Aggregated Wide-Stream Residual BiGRU
    EMBED_DIM = 32  # Dimension for atomic nucleotide and loop embeddings
    HIDDEN_DIM = 192  # 'H'. The BiGRU output/residual stream width W = 2 * H = 384
    NUM_LAYERS = 6  # Number of Wide-Stream Residual Blocks
    DROPOUT = 0.1  # Regularization

    # =========================================================================
    # Training Configuration
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 32  # Optimized for A100 usage with 107-length sequences
    EPOCHS = 25  # Sufficient for convergence with early stopping
    LEARNING_RATE = 1e-3  # Standard AdamW learning rate
    WEIGHT_DECAY = 1e-2  # Weight decay for AdamW
    MAX_GRAD_NORM = 1.0  # Gradient clipping threshold
    PATIENCE = 5  # Early stopping patience epochs

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers

    # Debugging / Development
    DEBUG = False  # If True, runs on a small subset of data
    SUBSET_SIZE = 128  # Number of samples to use in DEBUG mode
