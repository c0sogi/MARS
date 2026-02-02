import os
import torch


class Config:
    # --- General Configuration ---
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    exp_name = "idea_4"

    # --- Directories ---
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = f"./working/{exp_name}"

    # Ensure working directory exists
    os.makedirs(working_dir, exist_ok=True)

    # Data Paths
    train_metadata_path = os.path.join(metadata_dir, "train.csv")
    val_metadata_path = os.path.join(metadata_dir, "val.csv")
    test_metadata_path = os.path.join(metadata_dir, "test.csv")

    # The images are stored directly in train/ or test/ subfolders inside input
    # The metadata 'file_path' column already contains 'train/xxx.jpg' or 'test/xxx.jpg'
    image_base_dir = input_dir

    # --- Model Architecture ---
    # Using ConvNeXt-Small as the backbone
    model_name = "convnext_small.fb_in1k"
    pretrained = True
    num_classes = 11

    # Head Configuration
    use_gem_pooling = True  # Use Generalized Mean Pooling instead of standard GAP

    # --- Input Configuration ---
    image_size = 640  # Resolution set to 640x640
    in_chans = 3

    # --- Training Hyperparameters ---
    epochs = 12
    batch_size = 32  # Enforcing batch size >= 32
    num_workers = 12

    # --- Optimization ---
    lr = 5e-4
    min_lr = 1e-6
    weight_decay = 1e-2
    max_grad_norm = 10.0

    # --- Scheduler ---
    scheduler = "OneCycleLR"
    pct_start = 0.1
    div_factor = 25.0
    final_div_factor = 100.0

    # --- Regularization & Augmentation ---
    drop_rate = 0.0  # Dropout rate for the classifier head
    drop_path_rate = 0.2  # Stochastic depth rate for ConvNeXt

    # Augmentation Strength
    aug_prob = 0.75

    # --- Exponential Moving Average (EMA) ---
    use_ema = True
    ema_decay = 0.999  # Tuned decay rate for faster convergence

    # --- Hardware ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Target Columns ---
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
