import os
import torch


class Config:
    # ==========================================
    # Reproducibility & Environment
    # ==========================================
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4

    # ==========================================
    # File Paths
    # ==========================================
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_48"
    submission_dir = "./submission"

    # Metadata Sources
    train_metadata_path = os.path.join(metadata_dir, "train.parquet")
    val_metadata_path = os.path.join(metadata_dir, "val.parquet")
    test_metadata_path = os.path.join(metadata_dir, "test.parquet")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Cache Files (for deterministic data processing)
    train_cache_path = os.path.join(working_dir, "train_data_cache.npz")
    val_cache_path = os.path.join(working_dir, "val_data_cache.npz")
    test_cache_path = os.path.join(working_dir, "test_data_cache.npz")

    # Outputs
    model_save_path = os.path.join(working_dir, "best_model.pth")
    submission_file = os.path.join(submission_dir, "submission.csv")

    # ==========================================
    # Data Specifications
    # ==========================================
    seq_len = 107
    pred_len = 68

    # Input Feature Dimensions:
    # 4 (Nucleotides: A, G, C, U)
    # + 3 (Structure: (, ), .)
    # + 7 (Loop Type: S, M, I, B, H, E, X)
    input_dim = 14

    # Target Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    num_classes = 5

    # Scored Columns for MCRMSE (reactivity, deg_Mg_pH10, deg_Mg_50C)
    scored_classes_indices = [0, 1, 3]

    # ==========================================
    # Model Architecture
    # Strategy: Deep Stabilized Bias-Refined Decoupled BiGRU
    # ==========================================
    hidden_dim = 384  # High capacity
    n_layers = 4  # Deep backbone
    dropout = 0.1  # Regularization

    # Convolutional Stem
    conv_kernel_size = 3
    conv_filters = 256

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    batch_size = 64
    learning_rate = 1e-3
    weight_decay = 1e-2
    epochs = 50

    # Stability
    gradient_clip = 1.0  # Mandatory for 4-layer hybrid architecture

    # Optimization
    patience = 10  # Early stopping patience
    min_delta = 1e-4  # Minimum improvement for early stopping

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Set to a small integer (e.g., 100) to limit dataset size for rapid testing
    # Set to None for full training
    debug_samples = None

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)
