import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    debug_sample_size = 1000  # Number of samples to use when debug=True

    # -------------------------------------------------------------------------
    # Directories & Paths
    # -------------------------------------------------------------------------
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_3"

    # Ensure working directory exists
    os.makedirs(working_dir, exist_ok=True)

    # Metadata files
    train_metadata_path = os.path.join(metadata_dir, "train_metadata.csv")
    val_metadata_path = os.path.join(metadata_dir, "val_metadata.csv")
    test_metadata_path = os.path.join(metadata_dir, "test_metadata.csv")

    # Output paths
    model_save_path = os.path.join(working_dir, "best_model.pth")
    pos_weights_path = os.path.join(working_dir, "pos_weights.npy")
    submission_path = os.path.join(working_dir, "submission.csv")

    # -------------------------------------------------------------------------
    # Model Configuration
    # -------------------------------------------------------------------------
    model_name = "tf_efficientnetv2_s"
    pretrained = True
    image_size = 640
    in_channels = 3
    num_classes = 11

    # -------------------------------------------------------------------------
    # Training Configuration
    # -------------------------------------------------------------------------
    num_epochs = 12
    batch_size = 16
    learning_rate = 1e-4  # Constant learning rate
    weight_decay = 1e-2
    max_grad_norm = 10.0

    # Advanced Training
    use_amp = True
    use_ema = True
    ema_decay = 0.9999

    # Early Stopping
    patience = 3
    min_delta = 1e-4

    # -------------------------------------------------------------------------
    # Compute Configuration
    # -------------------------------------------------------------------------
    num_workers = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Target Columns
    # -------------------------------------------------------------------------
    target_cols = [
        "ETT - Abnormal",
        "ETT - Borderline",
        "ETT - Normal",
        "NGT - Abnormal",
        "NGT - Borderline",
        "NGT - Incompletely Imaged",
        "NGT - Normal",
        "CVC - Abnormal",
        "CVC - Borderline",
        "CVC - Normal",
        "Swan Ganz Catheter Present",
    ]
