import os
import torch


class Config:
    # --- General ---
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    num_workers = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    input_dir = "./input"
    train_metadata = "./metadata/train.csv"
    val_metadata = "./metadata/val.csv"
    test_metadata = "./metadata/test.csv"

    # Output directory for checkpoints and cache
    output_dir = "./working/idea_6"
    os.makedirs(output_dir, exist_ok=True)

    # Submission path
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    # --- Image Processing ---
    image_size = 640
    resize_mode = "longest_edge"  # Options: 'longest_edge', 'square'
    use_clahe = True
    clahe_clip_limit = 2.0
    clahe_tile_grid_size = (8, 8)

    # --- Model ---
    model_name = "convnextv2_tiny.fcmae_ft_in1k"
    # ConvNeXt V2 Tiny feature map channels: [96, 192, 384, 768]
    # We use Stage 3 (384) and Stage 4 (768) for Dual-Stage Pooling
    backbone_dim = 768 + 384
    pooling_stages = [3, 4]  # 0-indexed stages to extract from backbone

    drop_rate = 0.0  # Dropout rate for backbone
    drop_path_rate = 0.2  # Stochastic depth rate
    fc_dropout = 0.2  # Dropout for the final classification head

    # --- Training ---
    epochs = 10
    batch_size = 8  # Optimized for A100 with Tiny model
    learning_rate = 5e-4
    weight_decay = 1e-2
    max_grad_norm = 10.0

    # Scheduler
    scheduler_type = "OneCycleLR"
    pct_start = 0.1
    div_factor = 25.0
    final_div_factor = 100.0

    # EMA (Exponential Moving Average)
    use_ema = True
    ema_decay = 0.999  # Tuned for faster convergence

    # Loss
    pos_weight = 1.0  # Can be tuned for class imbalance

    # --- Targets ---
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
