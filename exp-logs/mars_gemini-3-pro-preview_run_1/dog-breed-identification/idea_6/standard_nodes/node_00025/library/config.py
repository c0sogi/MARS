import os
import random
import numpy as np
import torch


class Config:
    """
    Global configuration for the Dog Breed Classification pipeline.
    Includes settings for Data, Model, Training Strategy (Phases 1-3), and Inference.
    """

    # --------------------
    # General Settings
    # --------------------
    seed = 42
    debug = False  # Set to True to run with a small data subset for debugging
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------
    # Directory & File Paths
    # --------------------
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_6"
    submission_dir = "./submission"

    # Ensure output directories exist
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # Metadata Paths
    train_metadata_path = os.path.join(metadata_dir, "train.csv")
    val_metadata_path = os.path.join(metadata_dir, "val.csv")
    test_metadata_path = os.path.join(metadata_dir, "test.csv")

    # Submission Path
    submission_path = os.path.join(submission_dir, "submission.csv")

    # --------------------
    # Data Configuration
    # --------------------
    img_size = 256  # Input size for the model (H, W)
    resize_target = 274  # Initial resize dimension before CenterCrop
    batch_size = 16  # Batch size for training and inference
    num_classes = 120  # Total number of dog breeds

    # --------------------
    # Model Configuration
    # --------------------
    model_name = "convnext_base.fb_in1k"
    n_folds = 5  # Stratified K-Fold
    dropout_rate = 0.5  # Dropout for the classification head

    # --------------------
    # Training Strategy
    # --------------------
    # Phase 1: Head Adaptation (Frozen Backbone)
    head_adapt_epochs = 3
    head_adapt_lr = 1e-3

    # Phase 2: Fine-tuning (Unfrozen Backbone, Discriminative LRs)
    finetune_epochs = 7
    finetune_lr_backbone = 1e-6
    finetune_lr_head = 1e-4
    weight_decay = 1e-4

    # Phase 3: Stochastic Weight Averaging (SWA)
    swa_epochs = 5
    swa_lr = 5e-5
    # SWA starts after head adaptation and fine-tuning
    swa_start_epoch = head_adapt_epochs + finetune_epochs

    # --------------------
    # Inference Configuration
    # --------------------
    use_tta = True  # Enable Test Time Augmentation (Horizontal Flip)


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
