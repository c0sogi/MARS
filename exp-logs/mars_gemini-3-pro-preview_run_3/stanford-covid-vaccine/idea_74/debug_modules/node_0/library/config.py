import os
import torch


class Config:
    """
    Configuration class for the High-Capacity Topology-Aware Dual-Path BiGRU strategy.
    Centralizes all hyperparameters, file paths, and execution settings.
    """

    # ==========================================
    # General Settings
    # ==========================================
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 2  # For DataLoader

    # Debugging / Development
    # If debug is True, the data loader will only load a small subset of data
    debug = False
    debug_subset_size = 100

    # ==========================================
    # File Paths
    # ==========================================
    # Base directories
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_74"

    # Input Files (Metadata Parquet files)
    train_file = os.path.join(metadata_dir, "train.parquet")
    val_file = os.path.join(metadata_dir, "val.parquet")
    test_file = os.path.join(metadata_dir, "test.parquet")
    sample_submission_file = os.path.join(input_dir, "sample_submission.csv")

    # Cache Files (for deterministic data processing)
    # Using .npz for efficient storage of numpy arrays
    train_cache = os.path.join(working_dir, "train_data_cache.npz")
    val_cache = os.path.join(working_dir, "val_data_cache.npz")
    test_cache = os.path.join(working_dir, "test_data_cache.npz")

    # Model Checkpoints and Outputs
    best_model_path = os.path.join(working_dir, "best_model.pth")
    submission_path = os.path.join(working_dir, "submission.csv")

    # ==========================================
    # Model Architecture
    # ==========================================
    # Input features: 4 (Nucleotide One-Hot) + 3 (Structure One-Hot) + 7 (Loop Type One-Hot)
    input_dim = 14

    # High-Capacity Backbone (Lesson 103, Lesson 63)
    # Hidden dimension per direction. Total hidden size will be hidden_dim * 2
    hidden_dim = 384

    # Deep Architecture (Lesson 26)
    num_layers = 4

    # Regularization
    dropout = 0.1

    # Output
    num_classes = 5  # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    batch_size = 32
    epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-4  # Standard for AdamW

    # Optimization Stability (Lesson 46)
    max_grad_norm = 1.0

    # Scheduler (Cosine Annealing)
    T_max = 100  # Should match epochs usually
    eta_min = 1e-6

    # Early Stopping
    patience = 15

    # ==========================================
    # Data Specifications
    # ==========================================
    seq_len = 107
    pred_len = 68  # seq_scored

    # Columns to score for validation metric (Lesson 76)
    scored_columns = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # All target columns
    target_columns = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    def __init__(self):
        """
        Initialize configuration and ensure necessary directories exist.
        """
        os.makedirs(self.working_dir, exist_ok=True)
