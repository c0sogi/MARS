import os
import torch


class Config:
    # ==============================
    # General Configuration
    # ==============================
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4

    # ==============================
    # File Paths & Directories
    # ==============================
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_12"

    # Input Files (Metadata Parquet files)
    train_path = os.path.join(metadata_dir, "train.parquet")
    val_path = os.path.join(metadata_dir, "val.parquet")
    test_path = os.path.join(metadata_dir, "test.parquet")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Cache Paths (Hash-based caching will append hash to these base names in the pipeline)
    train_cache_base = os.path.join(working_dir, "train_data")
    val_cache_base = os.path.join(working_dir, "val_data")
    test_cache_base = os.path.join(working_dir, "test_data")

    # Output Paths
    model_save_path = os.path.join(working_dir, "best_model.pth")
    submission_path = os.path.join(working_dir, "submission.csv")

    # ==============================
    # Data Configuration
    # ==============================
    seq_len = 107
    pred_len = 68

    # Input Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
    input_channels = 14

    # Columns
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Only these are used for the competition metric, but we train on all 5
    scored_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # ==============================
    # Model Architecture
    # ==============================
    # Convolutional Stem
    conv_filters = 256
    kernel_size = 3

    # Latent Spatial Injection
    # Projecting concatenated [local, paired] vectors (256*2) back to hidden_dim

    # Backbone (BiGRU)
    hidden_dim = 384
    n_layers = 3
    dropout = 0.1

    # Output
    output_dim = 5  # Number of target columns

    # ==============================
    # Training Hyperparameters
    # ==============================
    batch_size = 64
    epochs = 50
    learning_rate = 1e-3
    weight_decay = 1e-4

    # Scheduler (Cosine Annealing)
    T_max = 50  # Should match epochs
    eta_min = 1e-6

    # Gradient Clipping
    max_grad_norm = 1.0

    # Early Stopping
    patience = 10

    def __init__(self):
        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

    def display(self):
        """Displays the configuration."""
        print("=" * 30)
        print("CONFIG")
        print("=" * 30)
        for attr in dir(self):
            if not attr.startswith("__") and not callable(getattr(self, attr)):
                print(f"{attr}: {getattr(self, attr)}")
        print("=" * 30)
