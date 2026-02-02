import os


class AudioConfig:
    """
    Configuration for audio processing and feature extraction.
    Defines the Multi-Resolution Log-Mel Spectrogram parameters.
    """

    sr = 16000
    duration = 1.0  # seconds
    n_samples = int(sr * duration)

    n_mels = 64
    hop_length = (
        160  # 10ms hop ensures consistent time dimension width (approx 101 frames)
    )

    # 3-Channel Multi-Resolution Settings
    # Format: (n_fft, win_length)
    # Channel 0: Short window (~20ms) for transient details
    # Channel 1: Medium window (~40ms) for balance
    # Channel 2: Long window (~60ms) for tonal/formant details
    resolutions = [
        (512, 320),  # 20ms * 16000
        (1024, 640),  # 40ms * 16000
        (2048, 960),  # 60ms * 16000
    ]

    fmin = 20
    fmax = 8000  # Nyquist
    top_db = 80.0


class ModelConfig:
    """
    Configuration for the Multi-Resolution SK-ResNet-Conformer architecture.
    """

    model_name = "skresnet34"
    pretrained = True
    num_classes = 12
    in_channels = 3  # Matches the 3 resolutions

    # Backbone modifications
    # Remove stride in last two stages to preserve temporal resolution
    # Output of SKResNet34 stage 4 is typically 512 channels
    backbone_out_dim = 512

    # Conformer Neck Parameters
    use_conformer = True
    conformer_dim = 512
    conformer_heads = 8
    conformer_layers = 2
    conformer_kernel_size = 31  # Large kernel for local context
    conformer_dropout = 0.1

    # Head Parameters
    pooling_type = "multi_head_attention"
    pooling_heads = 4


class TrainConfig:
    """
    Configuration for training, optimization, and file paths.
    """

    seed = 42
    debug = False
    debug_subset_size = 500  # Number of samples to use if debug is True

    # Optimization
    epochs = 15
    batch_size = 32
    lr = 1e-3
    weight_decay = 1e-2
    min_lr = 1e-6

    # Data Sampling
    use_weighted_sampler = True  # To handle class imbalance

    # Augmentation (SpecAugment)
    # Time mask: <20% of duration. Duration is ~100 frames, so param ~20.
    spec_aug_time_mask_param = 20
    spec_aug_freq_mask_param = 10
    spec_aug_min_val = -80.0  # Fill value for masks

    # Hardware
    num_workers = 4
    device = "cuda"

    # Paths
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Metadata files
    train_metadata_path = os.path.join(metadata_dir, "train.csv")
    val_metadata_path = os.path.join(metadata_dir, "val.csv")
    test_metadata_path = os.path.join(metadata_dir, "test.csv")

    # Working directory for caching and checkpoints
    working_dir = "./working/idea_9"
    cache_dir = os.path.join(working_dir, "cache")
    model_save_path = os.path.join(working_dir, "best_model.pth")
    submission_path = "./submission/submission.csv"

    @classmethod
    def setup_directories(cls):
        """Ensures working directories exist."""
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(cls.submission_path), exist_ok=True)
