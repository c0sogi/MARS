import os
import torch


class Config:
    # --- General Configuration ---
    seed = 42
    debug = False
    exp_name = "idea_5"
    working_dir = f"./working/{exp_name}"

    # --- Data Paths ---
    # Metadata files generated previously
    train_metadata = "./metadata/train.csv"
    val_metadata = "./metadata/val.csv"
    test_metadata = "./metadata/test.csv"

    # Image directories
    train_dir = "./input/train"
    test_dir = "./input/test"

    # --- Target Definition ---
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
    num_classes = len(target_cols)

    # --- Model Configuration ---
    # Backbone: ConvNeXt V2 Tiny (Pretrained on ImageNet-22k, FT on 1k)
    backbone = "convnextv2_tiny.fcmae_ft_in22k_in1k"
    pretrained = True
    in_chans = 3

    # Head Configuration
    use_gem_pooling = True
    use_multi_sample_dropout = True
    msd_num = 5  # Number of dropout masks
    msd_rate = 0.2  # Dropout rate for MSD

    # --- Input Configuration ---
    image_size = 640  # Resolution: 640x640

    # --- Training Hyperparameters ---
    epochs = 10
    batch_size = 32  # Stable batch size for Tiny backbone
    valid_batch_size = 64

    # Optimizer (AdamW)
    lr = 1e-3
    weight_decay = 1e-2
    max_grad_norm = 10.0

    # Scheduler (OneCycleLR)
    scheduler_type = "OneCycleLR"
    pct_start = 0.1
    div_factor = 25.0
    final_div_factor = 100.0

    # Model EMA (Exponential Moving Average)
    use_ema = True
    ema_decay = 0.999

    # Loss Function
    loss_fn = "BCEWithLogitsLoss"

    # --- Augmentation Parameters ---
    aug_prob = 0.75
    rotate_limit = 15
    scale_limit = 0.2
    shift_limit = 0.1
    coarse_dropout_holes = 8
    coarse_dropout_size = 32

    # --- Hardware ---
    num_workers = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Utility Methods ---
    @staticmethod
    def setup():
        """Creates the working directory if it does not exist."""
        os.makedirs(Config.working_dir, exist_ok=True)


# Execute setup on import
Config.setup()
