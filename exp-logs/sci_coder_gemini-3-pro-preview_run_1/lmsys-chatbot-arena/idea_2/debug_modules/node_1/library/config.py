import os
import torch


class Config:
    """
    Configuration class for the Siamese Transformer with Hybrid Feature Fusion model.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # ==========================
    # General Settings (Defaults)
    # ==========================
    seed = 42

    # ==========================
    # Data Paths
    # ==========================
    # Input metadata (read-only)
    train_path = "./metadata/train_metadata.csv"
    val_path = "./metadata/val_metadata.csv"
    test_path = "./metadata/test_metadata.csv"

    # Output paths
    working_dir = "./working/idea_2/"
    model_save_path = os.path.join(working_dir, "siamese_model.pth")

    # Submission
    submission_dir = "./submission/"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Cache files for features (parquet)
    train_features_path = os.path.join(working_dir, "train_scalar_features.parquet")
    val_features_path = os.path.join(working_dir, "val_scalar_features.parquet")
    test_features_path = os.path.join(working_dir, "test_scalar_features.parquet")

    # ==========================
    # Model Architecture
    # ==========================
    model_name = "microsoft/deberta-v3-small"
    max_length = 512
    num_classes = 3  # winner_model_a, winner_model_b, winner_tie

    # Scalar Features for Hybrid Fusion
    scalar_feature_cols = [
        "len_diff_char",
        "len_diff_word",
        "len_ratio_char",
        "len_ratio_word",
        "newline_diff",
        "newline_ratio",
    ]
    num_scalar_features = len(scalar_feature_cols)

    # ==========================
    # Training Hyperparameters
    # ==========================
    # Differential Learning Rates
    lr_backbone = 2e-5
    lr_head = 1e-3

    weight_decay = 0.01
    eps = 1e-6
    max_grad_norm = 1.0

    # Early Stopping
    patience = 2

    # ==========================
    # Hardware & Computation
    # ==========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4
    pin_memory = True

    def __init__(self, debug=False, epochs=4, train_batch_size=16, valid_batch_size=32):
        """
        Initialize configuration with optional overrides for flexibility.

        Args:
            debug (bool): If True, use a small subset of data for debugging.
            epochs (int): Number of training epochs.
            train_batch_size (int): Batch size for training.
            valid_batch_size (int): Batch size for validation/inference.
        """
        self.debug = debug
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size

        # Ensure necessary directories exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)
