import os
import random
import numpy as np
import torch


class Config:
    """
    Central configuration for the Tabular-Query Spatial-Attention Network (TQ-SAN).
    """

    # ==========================
    # General Settings
    # ==========================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================
    # File Paths
    # ==========================
    # Input Directories
    input_root = "./input"
    train_dir = os.path.join(input_root, "train")
    test_dir = os.path.join(input_root, "test")

    # Metadata Files (Generated in previous steps)
    metadata_dir = "./metadata"
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "val.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")
    sample_submission = os.path.join(input_root, "sample_submission.csv")

    # Working & Output Directories
    working_dir = "./working"
    idea_dir = os.path.join(working_dir, "idea_6")
    cache_dir = idea_dir  # For caching processed data
    model_save_path = os.path.join(idea_dir, "best_model.pth")
    submission_path = os.path.join(working_dir, "submission.csv")

    # ==========================
    # Data Hyperparameters
    # ==========================
    img_size = 224  # Native resolution for EfficientNet-B0
    in_channels = 3  # RGB (Tri-Slab MIPs)

    # TQ-SAN Specifics
    # We use 2 views: Axial and Coronal
    views = ["axial", "coronal"]

    # ==========================
    # Model Hyperparameters
    # ==========================
    backbone_name = "efficientnet_b0"
    pretrained = True

    # Tabular Feature Encoder
    n_tabular_features = 4  # Age, Sex, SmokingStatus, Baseline_Percent
    tabular_hidden_dim = 128

    # Attention Mechanism
    # The visual features from EfficientNet-B0 before pooling are 1280 channels
    visual_feature_dim = 1280
    projection_dim = 128  # Dimension for Query/Key/Value projections

    # Prediction Head
    dropout_rate = 0.2

    # ==========================
    # Training Hyperparameters
    # ==========================
    epochs = 30  # Short schedule as per strategy
    batch_size = 16  # Moderate batch size for multi-view 3D inputs

    # Optimizer (AdamW)
    lr = 1e-4
    weight_decay = 1e-2

    # Scheduler (Cosine Annealing)
    T_max = 30  # Matches epochs
    eta_min = 1e-6

    # Early Stopping
    patience = 8  # Stop if validation score doesn't improve

    # ==========================
    # Metric / Loss Constants
    # ==========================
    # Modified Laplace Log Likelihood constants
    min_sigma = 70.0  # Confidence clipping
    max_delta = 1000.0  # Error clipping

    # Quantile/Uncertainty settings for the model output
    # We predict: FVC_pred, Sigma_base, Sigma_growth

    @staticmethod
    def setup():
        """
        Sets up the environment:
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(Config.idea_dir, exist_ok=True)

        # Set seeds
        seed = Config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior for CuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Config setup complete. Device: {Config.device}, Seed: {Config.seed}")
        print(f"Cache directory: {Config.cache_dir}")
