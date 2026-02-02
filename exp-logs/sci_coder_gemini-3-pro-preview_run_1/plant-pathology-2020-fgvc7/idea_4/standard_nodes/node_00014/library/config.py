import os
import torch


class CFG:
    """
    Configuration class for Apple Disease Detection Task.
    Implements settings for Heterogeneous K-Fold Ensemble (ResNet34 + DenseNet121).
    """

    # ==========================================
    # General Settings
    # ==========================================
    seed = 42
    debug = False  # Set to True to run with a small subset for debugging

    # ==========================================
    # Directories & Paths
    # ==========================================
    input_root = "./input"
    images_dir = os.path.join(input_root, "images")

    metadata_dir = "./metadata"
    # We will load both train and val metadata and combine them for K-Fold CV
    train_metadata_path = os.path.join(metadata_dir, "train_metadata.csv")
    val_metadata_path = os.path.join(metadata_dir, "val_metadata.csv")
    test_metadata_path = os.path.join(metadata_dir, "test_metadata.csv")

    working_dir = "./working/idea_4"
    models_dir = os.path.join(working_dir, "models")

    # Submission
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Ensure working directories exist
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # ==========================================
    # Data Configuration
    # ==========================================
    img_size = 256
    num_classes = 4
    target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    n_folds = 5

    # ==========================================
    # Model Configuration
    # ==========================================
    # List of architectures for the heterogeneous ensemble
    model_architectures = ["resnet34", "densenet121"]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    batch_size = 32
    epochs = 15
    lr = 1e-4
    min_lr = 1e-6
    weight_decay = 1e-6

    # Scheduler (CosineAnnealingWarmRestarts)
    T_0 = 15
    T_mult = 1

    # ==========================================
    # Compute
    # ==========================================
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Inference
    # ==========================================
    tta_steps = 2  # 1 (Original) + 1 (Horizontal Flip)
