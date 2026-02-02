import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    project_name = "audio_tagging_convnext"
    experiment_name = "idea_8"
    seed = 42
    debug = False  # Set to True to run on a small subset for testing

    # =========================================================================
    # Directory Paths
    # =========================================================================
    # Read-only input directories
    input_root = "./input"
    train_audio_curated = os.path.join(input_root, "train_curated")
    train_audio_noisy = os.path.join(input_root, "train_noisy")
    test_audio = os.path.join(input_root, "test")

    # Metadata paths (pre-generated)
    metadata_root = "./metadata"
    train_csv = os.path.join(metadata_root, "train.csv")
    val_csv = os.path.join(metadata_root, "val.csv")
    test_csv = os.path.join(metadata_root, "test.csv")
    sample_submission = os.path.join(input_root, "sample_submission.csv")

    # Working directory for outputs (cache, checkpoints)
    working_dir = os.path.join("./working", experiment_name)
    os.makedirs(working_dir, exist_ok=True)

    # Output paths
    checkpoint_path = os.path.join(working_dir, "best_model.pth")
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    # =========================================================================
    # Audio Processing Parameters
    # =========================================================================
    sample_rate = 32000
    duration = 30.0  # seconds
    target_length = int(sample_rate * duration)

    # Spectrogram parameters
    n_fft = 1024
    hop_length = 512
    n_mels = 128
    f_min = 20
    f_max = 16000  # Nyquist frequency for 32kHz

    # Normalization
    # We use instance-level normalization, but these stats can be used for reference
    # or if we switch to dataset-level normalization later.
    norm_mean = 0.0
    norm_std = 1.0

    # =========================================================================
    # Model Architecture
    # =========================================================================
    backbone = "convnext_nano"
    pretrained = True
    in_channels = 1  # Single channel input (spectrogram)
    num_classes = 80

    # Pooling & Head
    pooling_type = (
        "dual_stream"  # Options: 'avg', 'max', 'dual_stream' (Attention + Max)
    )
    drop_rate = 0.2
    use_multi_sample_dropout = True
    multi_sample_dropout_count = 5

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 8  # Adjust based on CPU cores (12 vCPUs available)

    # Optimization
    epochs = 25
    batch_size = 32  # A100 40GB can handle larger, but 32 is safe for Nano + 30s audio
    learning_rate = 1e-3  # Max LR for OneCycle
    weight_decay = 1e-2

    # Scheduler (OneCycleLR)
    pct_start = 0.3
    div_factor = 25.0
    final_div_factor = 1000.0

    # Augmentation
    mixup_alpha = 0.2
    spec_augment_time_mask = 48
    spec_augment_freq_mask = 24

    # Early Stopping
    early_stopping_patience = 7
    early_stopping_metric = "lrap"  # 'loss' or 'lrap'
    early_stopping_mode = "max"  # 'min' for loss, 'max' for lrap

    # =========================================================================
    # Caching
    # =========================================================================
    use_cache = True  # Enable caching of processed spectrograms if implemented
