import os
import torch
import random
import numpy as np


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    project_name = "audio_tagging_idea_3"
    seed = 42
    debug = False  # Set to True to run with a small subset of data
    debug_sample_size = 200  # Number of samples to use in debug mode
    num_workers = 8  # Number of CPU workers for data loading

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input directories (Read-Only)
    input_root = "./input"
    train_curated_dir = os.path.join(input_root, "train_curated")
    train_noisy_dir = os.path.join(input_root, "train_noisy")
    test_dir = os.path.join(input_root, "test")

    # Metadata paths (Pre-generated)
    metadata_dir = "./metadata"
    train_csv_path = os.path.join(metadata_dir, "train.csv")
    val_csv_path = os.path.join(metadata_dir, "val.csv")
    test_csv_path = os.path.join(metadata_dir, "test.csv")
    sample_submission_path = os.path.join(input_root, "sample_submission.csv")

    # Output directories
    working_dir = "./working/idea_3"
    os.makedirs(working_dir, exist_ok=True)

    model_save_path = os.path.join(working_dir, "best_model.pth")

    # Submission directory
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    # =========================================================================
    # Audio Processing Parameters
    # =========================================================================
    sample_rate = 44100
    duration = 30  # seconds (Fixed duration strategy)
    target_length = sample_rate * duration  # 1,323,000 samples

    # Spectrogram conversion
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    fmin = 20
    fmax = 16000  # Approx Nyquist or specific cutoff

    # Augmentation
    mixup_alpha = 0.4
    spec_augment_freq_mask_param = 20
    spec_augment_time_mask_param = 40

    # =========================================================================
    # Model Architecture Parameters
    # =========================================================================
    # Backbone
    backbone_name = "efficientnet_b2"
    pretrained = True
    in_channels = 1  # Spectrogram has 1 channel

    # Transformer / Temporal Context
    # EfficientNet-B2 usually outputs 1408 channels at the last conv layer
    # We will project this to transformer_hidden_dim
    transformer_hidden_dim = 768
    transformer_layers = 2
    transformer_heads = 4
    transformer_dropout = 0.1

    # Head
    num_classes = 80
    pooling_heads = 4  # For Multi-Head Attention Pooling

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    epochs = 15
    batch_size = 32

    # Optimizer (AdamW)
    learning_rate = 1e-3
    weight_decay = 0.01

    # Scheduler (OneCycleLR)
    max_lr = 1e-3
    pct_start = 0.3
    div_factor = 25.0
    final_div_factor = 1000.0

    # Early Stopping
    patience = 5

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def set_seed(seed=42):
        """Sets the seed for reproducibility across various libraries."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
