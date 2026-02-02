import os
import torch


class CFG:
    """
    Configuration class for Diabetic Retinopathy Detection.
    Implements the 'Hybrid CNN-Transformer Ensemble' strategy (Idea 5).
    """

    # =====================
    # Meta Configuration
    # =====================
    seed = 42
    debug = False  # Set to True for fast debugging with smaller dataset
    debug_sample_size = 100  # Number of samples to use when debug=True
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =====================
    # File Paths
    # =====================
    # Input directories (Read-Only)
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Working directory (Read/Write)
    # Stores checkpoints, cached data, and logs
    working_dir = "./working/idea_5"
    os.makedirs(working_dir, exist_ok=True)

    # Metadata files
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "val.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")
    sample_submission_csv = os.path.join(input_dir, "sample_submission.csv")

    # Output paths
    submission_path = "./submission/submission.csv"
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # =====================
    # Model Architecture
    # =====================
    # General
    num_classes = 1  # Regression output (0-4 scale treated as continuous)
    pooling_type = "gem"  # Generalized Mean Pooling
    pretrained = True

    # Stream 1: CNN Backbone
    # EfficientNet-B5: Good balance of accuracy and compute for fine details
    model_cnn_name = "tf_efficientnet_b5_ns"
    img_size_cnn = 512

    # Stream 2: Transformer Backbone
    # Swin Transformer V2 Base: Captures long-range dependencies
    # Using the variant pre-trained for 384x384 resolution
    model_trans_name = "swinv2_base_window12to24_192to384_22kft1k"
    img_size_trans = 384

    # =====================
    # Training Hyperparameters
    # =====================
    n_folds = 5
    epochs = 15
    batch_size = 16  # Adjusted for A100 40GB with large models

    # Optimization
    lr = 1e-4
    weight_decay = 1e-5
    max_grad_norm = 10.0

    # Scheduler (Cosine Annealing)
    min_lr = 1e-6

    # Stochastic Weight Averaging (SWA)
    # Active in the final 20% of training
    swa_start_epoch_ratio = 0.8
    swa_lr = 1e-5  # Constant learning rate during SWA phase

    # =====================
    # Data Augmentation
    # =====================
    # Photometric
    clahe_prob = 0.5  # Stochastic CLAHE

    # Geometric
    flip_prob = 0.5
    rotation_degrees = 30

    # =====================
    # Inference
    # =====================
    use_tta = True  # Test Time Augmentation (Horizontal Flip)

    @classmethod
    def get_swa_start_epoch(cls):
        """Calculates the epoch to start SWA based on the ratio."""
        return int(cls.epochs * cls.swa_start_epoch_ratio)
