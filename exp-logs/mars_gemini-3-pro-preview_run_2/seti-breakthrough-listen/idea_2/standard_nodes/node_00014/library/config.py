import os
import torch


class Config:
    # =========================================================================
    # General Configuration
    # =========================================================================
    seed = 42
    debug = False  # Set to True to use a small subset of data for debugging
    debug_sample_size = 1000  # Number of samples to use when debug=True

    # =========================================================================
    # Path Configuration
    # =========================================================================
    # Root directories
    input_root = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working"

    # Specific output directories
    # We use 'idea_2' to store artifacts for this specific experimental run
    output_dir = os.path.join(working_dir, "idea_2")
    submission_dir = "./submission"

    # Metadata files (Pre-generated)
    train_csv_path = os.path.join(metadata_dir, "train.csv")
    val_csv_path = os.path.join(metadata_dir, "val.csv")
    test_csv_path = os.path.join(metadata_dir, "test.csv")

    # Raw data paths
    sample_submission_path = os.path.join(input_root, "sample_submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Spectrogram dimensions
    # The input consists of 6 panels (ABACAD cadence), each 273x256.
    # We vertically stack them: Total Height = 6 * 273 = 1638.
    height = 1638
    width = 256
    image_size = (height, width)

    # Input channels
    # Raw data is single-channel, but we expand to 3 channels to utilize
    # pretrained ImageNet weights effectively.
    in_channels = 3

    # =========================================================================
    # Model Configuration
    # =========================================================================
    # Backbone: EfficientNet-V2 Small
    model_name = "tf_efficientnetv2_s"
    pretrained = True
    num_classes = 1

    # Generalized Mean Pooling (GeM) Settings
    # Replaces standard Global Average Pooling to better detect sparse signals
    use_gem = True
    gem_p = 3.0  # Initial power parameter (p > 1 approximates Max Pooling)
    gem_trainable = True

    # Regularization within model
    drop_rate = 0.2  # Dropout rate for the classifier head
    drop_path_rate = 0.2  # Stochastic depth rate for the backbone

    # =========================================================================
    # Training Configuration
    # =========================================================================
    epochs = 15
    batch_size = 16  # Reduced to prevent OOM
    num_workers = 8  # Utilization of available vCPUs

    # Optimizer (AdamW)
    lr = 1e-4
    weight_decay = 1e-2

    # Scheduler (CosineAnnealingLR)
    min_lr = 1e-6

    # Loss Function
    # We use BCEWithLogitsLoss for binary classification

    # =========================================================================
    # Augmentation Configuration
    # =========================================================================
    # Mixup: Blends images and labels to improve robustness against noise
    use_mixup = True
    mixup_alpha = 1.0

    # Spatial Augmentations
    hflip_prob = 0.5  # Horizontal flip (Frequency axis) - Critical for Doppler drift
    vflip_prob = 0.5  # Vertical flip (Time axis)

    # =========================================================================
    # Hardware
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Utilities
    # =========================================================================
    print_freq = 50  # Frequency of logging during training loops

    @classmethod
    def create_dirs(cls):
        """Creates the necessary working and submission directories."""
        os.makedirs(cls.output_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)
