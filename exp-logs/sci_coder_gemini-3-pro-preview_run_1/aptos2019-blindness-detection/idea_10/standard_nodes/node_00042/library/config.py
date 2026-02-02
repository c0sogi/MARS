import os
import random
import numpy as np
import torch


class Config:
    # === Experiment Setup ===
    seed = 42
    exp_name = "idea_11"
    debug = False  # Toggle to True to run on a small subset for debugging

    # === Directories ===
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = f"./working/{exp_name}"

    # Ensure working directory exists
    os.makedirs(working_dir, exist_ok=True)

    # === Data Parameters ===
    image_size = 640
    num_classes = 5  # Labels: 0, 1, 2, 3, 4

    # Metadata File Paths
    train_metadata_path = os.path.join(metadata_dir, "train.csv")
    val_metadata_path = os.path.join(metadata_dir, "val.csv")
    test_metadata_path = os.path.join(metadata_dir, "test.csv")

    # === Model Architecture ===
    # Using ConvNeXt Base pretrained on ImageNet-22k and finetuned on 1k
    backbone = "convnext_base.fb_in22k_ft_in1k"
    pretrained = True

    # Regularization within Backbone
    drop_rate = 0.0  # Classifier dropout (not used given custom head)
    drop_path_rate = 0.4  # Stochastic depth rate

    # Custom Head Configuration
    use_multi_scale = True  # Aggregates features from Stage 3 and Stage 4
    head_dropout = 0.2  # Dropout rate in the custom ordinal head
    num_ordinal_heads = (
        4  # 4 binary classifiers for ordinal regression (0, 1, 2, 3 thresholds)
    )

    # === Training Hyperparameters ===
    batch_size = 16  # Optimized for A100 40GB with 640x640 resolution
    epochs = 15  # Sufficient for fine-tuning

    # Optimizer (AdamW)
    lr = 1e-4  # Initial learning rate
    min_lr = 1e-6  # Minimum learning rate for scheduler
    weight_decay = 0.05  # High weight decay for ConvNeXt
    max_grad_norm = 10.0  # Gradient clipping threshold
    accumulate_grad_batches = 1

    # Scheduler
    T_max = epochs  # Cosine annealing period

    # === Augmentation & Regularization ===
    # Mixup
    mixup_alpha = 0.4  # Beta distribution parameter
    mixup_prob = 1.0  # Probability of applying mixup

    # Geometric Augmentations (No photometric/hue changes)
    # Handled in dataset class, but flagged here for reference
    aug_hflip_prob = 0.5
    aug_vflip_prob = 0.5
    aug_rotate90_prob = 0.5

    # === Exponential Moving Average (EMA) ===
    use_ema = True
    ema_decay = 0.999  # Decay rate for shadow weights

    # === Inference ===
    tta_steps = 4  # 4-View TTA: Original, HFlip, VFlip, Rotate180

    # === Hardware & Logging ===
    num_workers = 8  # Number of dataloader workers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_freq = 10  # Logging frequency in steps


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = (
        False  # False ensures reproducibility at cost of some speed
    )
