import os
import torch


class Config:
    """
    Central configuration for the ResNet18-D Multi-Task U-Net pipeline.
    Handles hyperparameters, file paths, and system settings.
    """

    # =========================
    # General & System
    # =========================
    seed = 42
    debug = False  # Set to True to limit dataset size for debugging
    debug_sample_size = 100  # Number of samples to use when debug=True
    num_workers = 4  # Optimized for the available vCPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # Data Paths
    # =========================
    input_dir = "./input"

    # Pre-generated metadata paths
    train_metadata_path = "./metadata/train.csv"
    val_metadata_path = "./metadata/val.csv"
    test_metadata_path = "./metadata/test.csv"

    # Caching and Output
    # Directory for caching deterministic data processing (e.g. numpy arrays)
    cache_dir = "./working/idea_12"

    # Submission output
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Ensure necessary writeable directories exist
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # =========================
    # Model Architecture
    # =========================
    backbone = "resnet18d"  # Deep stem ResNet variant
    img_size = 512
    num_study_classes = 4

    # Class labels for reference
    study_labels = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]

    # =========================
    # Training Hyperparameters
    # =========================
    epochs = 20
    train_batch_size = 32
    valid_batch_size = 32

    # Optimization
    # Learning rate set based on batch size of 32 using linear scaling principles suitable for AdamW
    learning_rate = 5e-4
    weight_decay = 1e-2

    # Scheduler (Cosine Annealing)
    min_lr = 1e-6

    # Loss Function Weights
    # Prioritize segmentation to force spatial feature learning
    lambda_cls = 1.0
    lambda_seg = 10.0
