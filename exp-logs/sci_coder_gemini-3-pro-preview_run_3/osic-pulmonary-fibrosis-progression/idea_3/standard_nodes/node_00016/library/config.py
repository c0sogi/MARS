import os
import torch


class Config:
    """
    Configuration for the Content-Adaptive 2.5D Multi-View Network experiment (Idea 3).
    """

    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    seed = 42
    debug = False
    exp_name = "idea_3"

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = f"./working/{exp_name}"

    # Input Metadata Files
    train_csv_path = os.path.join(metadata_dir, "train.csv")
    val_csv_path = os.path.join(metadata_dir, "val.csv")
    test_csv_path = os.path.join(metadata_dir, "test.csv")

    # Output Directories
    # Used for caching processed 2.5D slice tensors
    cache_dir = os.path.join(working_dir, "cache")
    # Used for saving model weights
    checkpoint_dir = os.path.join(working_dir, "checkpoints")
    # Final submission file
    submission_path = "./submission/submission.csv"

    # Create necessary directories immediately
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Parameters
    # -------------------------------------------------------------------------
    img_size = 256  # Resize DICOM slices to 256x256
    num_slices = 3  # 3 views: Apical, Middle, Basal

    # Normalization Constants (Derived from EDA)
    # Used for Z-score standardization of the target FVC
    target_mean = 2654.6528
    target_std = 801.7017

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    backbone_name = "efficientnet_b0"
    pretrained = True
    freeze_backbone = True  # Backbone is frozen as per strategy

    # Feature Dimensions
    # EfficientNet-B0 outputs 1280 features.
    # Combined input = 3 slices * 1280 = 3840.
    slice_feature_dim = 1280
    combined_feature_dim = slice_feature_dim * num_slices

    # Projection Head
    # Project high-dim visual features down to prevent overwhelming tabular data
    projection_dim = 128

    # Tabular Branch
    # Inputs: Age, Sex, SmokingStatus, Weeks, Baseline_FVC
    tabular_input_dim = 5
    tabular_hidden_dim = 64

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    epochs = 50  # Extended training duration for scheduler convergence
    batch_size = 32
    learning_rate = 1e-3
    weight_decay = 1e-5
    patience = 10  # Early stopping patience

    # Metric / Loss Constants
    sigma_clip = 70.0  # Minimum confidence value for metric calculation
    max_error = 1000.0  # Error threshold for metric calculation

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
