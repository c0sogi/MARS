import os
import torch


class Config:
    # General Settings
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging

    # Data Settings
    input_dir = "./input"
    metadata_dir = "./metadata"
    train_metadata = os.path.join(metadata_dir, "train.csv")
    val_metadata = os.path.join(
        metadata_dir, "val.csv"
    )  # Note: We use 5-fold CV on the full train set usually, but complying with metadata structure.
    # Actually, the strategy says "The training data will be divided into 5 stratified folds".
    # We will likely merge train.csv and val.csv in the training script or just use train.csv and split it.
    # Given the prompt provided metadata files, we will point to them.
    test_metadata = os.path.join(metadata_dir, "test.csv")

    # Output Settings
    working_dir = "./working/idea_5"
    checkpoint_dir = os.path.join(working_dir, "checkpoints")
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Model Architecture
    model_name = "convnext_small"  # timm model name
    pretrained = True
    num_classes = 1  # Binary classification (Dog vs Cat)

    # Input Parameters
    image_size = 224
    input_channels = 3

    # Training Hyperparameters
    n_folds = 5
    epochs = 15
    batch_size = 64
    num_workers = 4  # 12 vCPUs available, 4 is a safe number

    # Optimization
    learning_rate = 1e-4
    weight_decay = 1e-2
    min_lr = 1e-6

    # Augmentation
    mixup_alpha = 0.2
    min_crop_scale = 0.8  # RandomResizedCrop minimum scale

    # Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Creates necessary directories."""
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.checkpoint_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)


# Automatically setup directories when imported
Config.setup()
