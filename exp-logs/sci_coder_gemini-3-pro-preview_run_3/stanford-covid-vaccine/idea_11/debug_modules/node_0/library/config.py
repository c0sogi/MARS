import os
import torch


class Config:
    """
    Configuration class for the RNA Degradation Prediction task.
    Implements the 'Latent Spatially-Augmented BiGRU' strategy settings.
    """

    # ==============================
    # General Settings
    # ==============================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4  # Number of dataloader workers

    # ==============================
    # File Paths & Directories
    # ==============================
    # Base directories
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_11"

    # Ensure working directory exists
    os.makedirs(working_dir, exist_ok=True)

    # Data paths (using pre-generated metadata Parquet files)
    train_path = os.path.join(metadata_dir, "train.parquet")
    val_path = os.path.join(metadata_dir, "val.parquet")
    test_path = os.path.join(metadata_dir, "test.parquet")

    # Sample submission for format reference
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Output paths
    # Cache file for preprocessed tensors (hash-based naming handled in processing module)
    cache_dir = working_dir
    model_save_path = os.path.join(working_dir, "best_model.pth")
    submission_path = os.path.join(working_dir, "submission.csv")

    # ==============================
    # Data Dimensions & Features
    # ==============================
    seq_len = 107
    pred_len = 68  # Number of positions scored

    # Feature Vocabularies (One-Hot Encoding)
    # Sequence: A, G, U, C
    vocab_seq = 4
    # Structure: (, ), .
    vocab_struct = 3
    # Predicted Loop Type: S, M, I, B, H, E, X
    vocab_loop = 7

    # Total input channels = 4 + 3 + 7 = 14
    input_channels = vocab_seq + vocab_struct + vocab_loop

    # Targets
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    num_targets = 5

    # ==============================
    # Model Hyperparameters
    # ==============================
    # Convolutional Stem (Local Feature Extraction)
    conv_filters = 256
    conv_kernel = 3

    # Backbone (High-Capacity BiGRU)
    # Strategy: Maximize capacity over hierarchical depth
    rnn_hidden_dim = 384
    rnn_layers = 3
    dropout = 0.1  # Standard dropout for RNN regularization

    # Latent Spatial Mixing
    # This logic is handled in the model definition, but relies on structure info

    # ==============================
    # Training Hyperparameters
    # ==============================
    epochs = 50
    batch_size = 64

    # Optimizer (AdamW)
    lr = 1e-3
    weight_decay = 1e-4

    # Gradient Clipping (Critical for high-capacity RNN stability)
    clip_grad_norm = 1.0

    # Scheduler (Cosine Annealing)
    T_max = 50  # Should match epochs
    eta_min = 1e-6

    # Early Stopping
    patience = 10

    def __init__(self, **kwargs):
        """
        Allow overriding configuration parameters during instantiation.
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
