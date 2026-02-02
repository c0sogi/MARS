import os
import torch


class Config:
    # =======================
    # General Settings
    # =======================
    project_name = "artwork_attribution"
    run_name = "idea_5"
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    debug_sample_size = 2000
    num_workers = 4

    # =======================
    # Directories & Paths
    # =======================
    # Input paths (Read-Only)
    input_dir = "./input"
    train_metadata_path = "./metadata/train.csv"
    val_metadata_path = "./metadata/val.csv"
    test_metadata_path = "./metadata/test.csv"

    # Output paths (Working Directory)
    output_dir = os.path.join("./working", run_name)
    model_save_path = os.path.join(output_dir, "convnext_small_best.pth")
    submission_path = os.path.join(output_dir, "submission.csv")

    # =======================
    # Model Architecture
    # =======================
    model_name = "convnext_small"  # timm backbone
    num_classes = 3474
    pretrained = True
    image_size = 320

    # Pooling
    pooling_type = "gem"  # Generalized Mean Pooling

    # Model EMA (Exponential Moving Average)
    use_ema = True
    ema_decay = 0.9999

    # =======================
    # Training Hyperparameters
    # =======================
    epochs = 20
    batch_size = 32  # Adjusted for A100 40GB memory with 320x320 resolution

    # Optimizer (AdamW)
    learning_rate = 1e-3
    weight_decay = 1e-2
    min_lr = 1e-6  # For Cosine Annealing

    # Loss Function
    # Moderate positive weight to handle class imbalance without precision collapse
    pos_weight = 10.0

    # Label Smoothing to prevent overconfidence on noisy labels
    label_smoothing = 0.05

    # =======================
    # Augmentation & Regularization
    # =======================
    # Mixup and CutMix parameters
    mixup_alpha = 0.4
    cutmix_alpha = 1.0
    mixup_prob = 0.5  # Probability of applying Mixup/CutMix batch-wise

    # =======================
    # Inference
    # =======================
    # Threshold tuning range for F1 score maximization
    threshold_start = 0.01
    threshold_end = 0.99
    threshold_step = 0.01

    # =======================
    # Hardware
    # =======================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = True

    @classmethod
    def setup(cls):
        """Ensures the output directory exists."""
        os.makedirs(cls.output_dir, exist_ok=True)
