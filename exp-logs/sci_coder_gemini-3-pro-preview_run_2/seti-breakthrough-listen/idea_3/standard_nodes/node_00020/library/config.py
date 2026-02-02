import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # General Settings
    # --------------------------------------------------------------------------
    seed = 42
    debug = False  # Set to True to run with a small subset of data for debugging
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_freq = 100  # Frequency of logging during training

    # --------------------------------------------------------------------------
    # Directories and Paths
    # --------------------------------------------------------------------------
    input_root = "./input"
    train_dir = os.path.join(input_root, "train")
    test_dir = os.path.join(input_root, "test")

    metadata_dir = "./metadata"
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "val.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")

    # Output directory for Idea 3
    output_dir = "./working/idea_3"
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "best_model.pth")
    submission_path = os.path.join(output_dir, "submission.csv")

    # --------------------------------------------------------------------------
    # Data & Image Parameters
    # --------------------------------------------------------------------------
    # The input data comes in shape (6, 273, 256).
    # We stack the 6 panels vertically: 6 * 273 = 1638.
    img_height = 1638
    img_width = 256

    # Model expects 3 channels (RGB), so we will replicate the single channel
    in_channels = 3

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    model_name = "convnext_tiny"
    pretrained = True

    # Dropout rates
    drop_rate = 0.0  # Head dropout
    drop_path_rate = 0.1  # Stochastic depth rate for ConvNeXt

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    epochs = 12
    batch_size = 16  # Reduced to fit 16GB VRAM

    # Optimizer (AdamW)
    lr = 1e-4
    weight_decay = 1e-2
    max_grad_norm = 10.0

    # Scheduler (Cosine Annealing Warm Restarts)
    T_0 = epochs  # Cycle length
    T_mult = 1
    min_lr = 1e-6

    # --------------------------------------------------------------------------
    # Augmentation & Regularization
    # --------------------------------------------------------------------------
    # Mixup
    mixup_alpha = 1.0
    mixup_prob = 0.5

    # CoarseDropout (Cutout)
    # Applied via Albumentations in the dataset class
    coarse_dropout_prob = 0.25
    coarse_dropout_num_holes_min = 4
    coarse_dropout_num_holes_max = 12
    coarse_dropout_hole_height = 64
    coarse_dropout_hole_width = 64

    # --------------------------------------------------------------------------
    # Inference
    # --------------------------------------------------------------------------
    tta = True  # Enable Test Time Augmentation (Horizontal Flip)
