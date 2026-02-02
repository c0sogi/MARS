import os
import torch


class Config:
    # --- General ---
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging

    # --- Directories ---
    # Input data is read-only
    input_dir = "./input"
    # Metadata is pre-generated
    metadata_dir = "./metadata"
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "val.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")

    # Working directory for this specific idea/experiment
    # We use idea_3 as per the working directory info provided
    working_dir = "./working/idea_3"
    os.makedirs(working_dir, exist_ok=True)

    # Output paths
    model_save_path = os.path.join(working_dir, "resnet101_best.pth")
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    # --- Data ---
    # Full dataset usage is enforced unless debug is True
    img_size = 320
    num_classes = 3474
    # A100 40GB can handle larger batches.
    # ResNet101 @ 320x320: 128 is usually safe with AMP.
    batch_size = 128
    num_workers = 12  # Matches available vCPUs

    # --- Model ---
    model_name = "resnet101"
    pretrained = True

    # --- Training ---
    epochs = 15
    lr = 1e-3
    weight_decay = 1e-2

    # Asymmetric Loss (ASL) Hyperparameters
    # Designed to handle high imbalance by down-weighting easy negatives
    asl_gamma_neg = 4.0
    asl_gamma_pos = 1.0
    asl_clip = 0.05

    # --- Inference ---
    # Threshold for converting probabilities to labels
    # Will be calibrated on validation set, but this is a default start
    base_threshold = 0.5

    # --- Hardware ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __str__(self):
        """Prints the configuration."""
        attributes = [
            attr
            for attr in dir(self)
            if not attr.startswith("__") and not callable(getattr(self, attr))
        ]
        return "\n".join([f"{attr}: {getattr(self, attr)}" for attr in attributes])
