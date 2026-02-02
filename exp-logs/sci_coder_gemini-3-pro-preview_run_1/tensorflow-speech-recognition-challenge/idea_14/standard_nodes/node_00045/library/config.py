import os
import torch


class Config:
    """
    Central configuration for the Speech Command Recognition task.
    Implements the settings for the Self-Distillation (Born-Again Networks) strategy.
    """

    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    seed = 42
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    input_dir = "./input"
    train_audio_dir = os.path.join(input_dir, "train", "audio")
    test_audio_dir = os.path.join(input_dir, "test", "audio")

    metadata_dir = "./metadata"
    train_metadata_path = os.path.join(metadata_dir, "train.csv")
    val_metadata_path = os.path.join(metadata_dir, "val.csv")
    test_metadata_path = os.path.join(metadata_dir, "test.csv")

    # Working directory for the specific idea (Idea 14: Self-Distillation)
    working_dir = "./working/idea_14"
    cache_dir = working_dir
    checkpoint_dir = working_dir

    # Output for submission
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # -------------------------------------------------------------------------
    # Audio Parameters
    # -------------------------------------------------------------------------
    sample_rate = 16000
    duration = 1.0  # seconds

    # Spectrogram extraction
    n_mels = 128
    n_fft = 1024
    hop_length = 160  # 10ms at 16kHz
    f_min = 0
    f_max = None  # Defaults to sample_rate // 2

    # -------------------------------------------------------------------------
    # Model Parameters
    # -------------------------------------------------------------------------
    backbone = "efficientnet_b2"
    in_channels = 1  # Adapted from RGB weights
    # 35 classes covers: 10 targets + 1 silence + ~20 auxiliary commands + buffer
    num_classes = 35

    # -------------------------------------------------------------------------
    # Training Parameters
    # -------------------------------------------------------------------------
    # Two-stage training: Teacher (Standard) -> Student (Distilled)
    epochs_teacher = 40
    epochs_student = 40

    batch_size = 128  # Optimized for A100 GPU
    learning_rate = 1e-3
    weight_decay = 1e-2

    # Optimizer & Scheduler
    optimizer = "AdamW"
    scheduler = "CosineAnnealingLR"
    min_lr = 1e-6

    # -------------------------------------------------------------------------
    # Augmentation & Regularization
    # -------------------------------------------------------------------------
    mixup_alpha = 1.0
    distillation_lambda = 0.5  # Weight for KL Divergence loss in Stage 2

    # Waveform augmentations
    noise_snr_min = 10  # dB
    noise_snr_max = 30  # dB

    # SpecAugment parameters
    freq_mask_param = 15
    time_mask_param = 35

    # -------------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------------
    target_labels = [
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
    ]
    silence_label = "silence"
    unknown_label = "unknown"

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    debug = False
    debug_sample_size = 200  # Number of samples to use when debug=True

    @classmethod
    def create_dirs(cls):
        """Helper to create necessary directories."""
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)
