import os
import torch


class CFG:
    """
    Configuration class for Vesuvius Ink Detection.
    Implements parameters for the Wide-Context SegFormer (MiT-B4) with Learnable Z-Compression.
    """

    # ====================================================
    # General Settings
    # ====================================================
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 12  # Optimized for the 12 vCPUs available

    # ====================================================
    # Directories & Paths
    # ====================================================
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_9"

    # Ensure the working directory exists for caching and checkpoints
    os.makedirs(working_dir, exist_ok=True)

    # Metadata file paths
    train_metadata_path = os.path.join(metadata_dir, "train.csv")
    val_metadata_path = os.path.join(metadata_dir, "validation.csv")
    test_metadata_path = os.path.join(metadata_dir, "test.csv")

    # Output paths
    model_path = os.path.join(working_dir, "best_model.pth")
    submission_path = "./submission.csv"  # Required output location

    # ====================================================
    # Data & Preprocessing
    # ====================================================
    image_size = 512

    # 5 Overlapping Thick Slabs Configuration
    # Each tuple represents (start_slice, end_slice) relative to the volume
    # Channel 1: 14-26
    # Channel 2: 20-32
    # Channel 3: 26-38 (Center)
    # Channel 4: 32-44
    # Channel 5: 38-50
    z_ranges = [(14, 26), (20, 32), (26, 38), (32, 44), (38, 50)]

    input_channels = len(z_ranges)  # 5

    # ====================================================
    # Augmentation
    # ====================================================
    # Volumetric Z-Jitter: randomly shift the start index of the 5-slab window
    z_jitter_range = 2  # +/- 2 slices

    # ====================================================
    # Model Architecture
    # ====================================================
    model_name = "nvidia/mit-b4"
    num_classes = 1

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    epochs = 15
    batch_size = 8  # A100-40GB can handle batch size 8 for MiT-B4 @ 512x512
    learning_rate = 1e-4
    weight_decay = 1e-2

    # Learning Rate Scheduler (ReduceLROnPlateau)
    scheduler_patience = 3
    scheduler_factor = 0.5
    min_lr = 1e-6

    # ====================================================
    # Evaluation & Inference
    # ====================================================
    # Metric: F0.5 Score (Beta=0.5)
    beta = 0.5

    # Binary classification threshold
    threshold = 0.5

    # Validation Gating: Only generate submission if val score > baseline
    baseline_score = 0.598
