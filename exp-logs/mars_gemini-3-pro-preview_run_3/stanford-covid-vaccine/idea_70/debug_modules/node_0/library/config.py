import os
import torch


class Config:
    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_70"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Files (Parquet format)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")

    # Reference Files
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache Files (NPZ format for efficient array storage)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_cache.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data_cache.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data_cache.npz")

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Model Hyperparameters
    # Strategy: High-Capacity GLU-Refined Decoupled BiGRU
    # =========================================================================
    # Input Features:
    #   - Nucleotide (A, G, C, U) -> 4
    #   - Structure (., (, )) -> 3
    #   - Loop Type (S, M, I, B, H, E, X) -> 7
    #   Total Channels = 14
    input_dim = 14

    # Convolutional Stem
    stem_channels = 256
    stem_kernel_size = 3

    # Backbone: 4-Layer Bidirectional GRU
    # Hidden dimension 384 per direction results in 768 total features per step
    hidden_dim = 384
    num_layers = 4

    # Interaction Module (GLU-Decoupled)
    # Wide projection for the gate mechanism to avoid bottlenecks
    gate_hidden_dim = 384

    # Regularization
    dropout = 0.1

    # Output
    num_targets = 5

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4

    # Optimization
    learning_rate = 1e-3
    weight_decay = 1e-4  # Standard for AdamW
    batch_size = 32  # Optimized for A100 memory
    epochs = 50
    max_grad_norm = 1.0  # Mandatory gradient clipping

    # Scheduler (Cosine Annealing)
    T_max = 50
    eta_min = 1e-6

    # Early Stopping
    patience = 10

    # =========================================================================
    # Data & Scoring
    # =========================================================================
    seq_length = 107
    seq_scored = 68

    # Target Columns
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Scoring Columns (Subset of targets used for metric calculation)
    scored_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Debugging / Development
    # If debug is True, only a small subset of data is used for rapid iteration
    debug = False
    debug_subset_size = 100
