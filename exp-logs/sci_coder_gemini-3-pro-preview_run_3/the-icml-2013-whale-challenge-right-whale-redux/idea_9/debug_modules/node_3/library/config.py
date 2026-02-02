import os
import torch


class AudioConfig:
    """
    Configuration for Audio Processing.
    Implements Compound Scaling and High-Resolution Time-Frequency conversion.
    """

    # Sampling rate from data analysis (2000 Hz)
    sample_rate = 2000

    # Duration in seconds (max duration is 2.0s)
    duration = 2.0

    # Spectrogram Parameters
    # Compound Scaling: Increased vertical resolution to match ConvNeXt capacity
    n_mels = 320

    # FFT Window size: 1024 gives 513 freq bins, sufficient for 320 mels
    n_fft = 1024

    # Hop length: 20 samples @ 2000Hz = 10ms (<15ms requirement for transient fidelity)
    hop_length = 20

    # Frequency range
    fmin = 15
    fmax = 1000  # Nyquist frequency

    # Normalization Strategy
    # Structural Innovation: Frequency-Wise Standardization to remove stationary noise
    freq_wise_standardization = True


class ModelConfig:
    """
    Configuration for the Neural Network Architecture.
    """

    # Backbone: ConvNeXt-Small (larger capacity than Tiny/B2)
    # Replaced specific tag 'convnext_small.in12k_ft_in1k' with generic 'convnext_small'
    # to avoid "Unknown model" error in the installed timm version.
    model_name = "convnext_small"
    pretrained = True

    # Input channels (1 for spectrogram)
    in_chans = 1

    # Classification head
    num_classes = 1

    # Structural Innovation: Coordinate Attention
    # Factorizes attention into time/freq axes within the backbone
    use_coordinate_attention = True

    # Pooling: Generalized Mean Pooling to focus on transient calls
    pooling_type = "gem"

    # Regularization
    drop_rate = 0.3
    drop_path_rate = 0.2


class TrainConfig:
    """
    Configuration for Training, Optimization, and Augmentation.
    """

    # Reproducibility
    seed = 42

    # Directories
    input_dir = "./input"
    train_dir = os.path.join(input_dir, "train2")
    test_dir = os.path.join(input_dir, "test2")

    # Metadata
    metadata_dir = "./metadata"
    train_meta = os.path.join(metadata_dir, "train.csv")
    val_meta = os.path.join(metadata_dir, "val.csv")
    test_meta = os.path.join(metadata_dir, "test.csv")

    # Working Directory for Caching and Models
    working_dir = "./working/idea_9"
    os.makedirs(working_dir, exist_ok=True)

    # Training Hyperparameters
    batch_size = 32
    epochs = 20  # Early stopping will likely trigger before this

    # Optimization
    # AdamW with Cosine Annealing
    optimizer_name = "AdamW"
    lr = 3e-4
    min_lr = 1e-6
    weight_decay = 0.05
    scheduler_name = "CosineAnnealingLR"

    # Loss Function Handling
    # Inverse Class Frequency Weighting: Neg ~16340, Pos ~1813 -> Ratio ~9
    bce_pos_weight = 9.0

    # Augmentation
    # Mixup with Mixed Losses (scalar loss mixing)
    use_mixup = True
    mixup_alpha = 0.4
    mixup_prob = 0.5

    # SpecAugment
    use_spec_augment = True
    freq_mask_param = 40
    time_mask_param = 60

    # Hardware
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging flag (set to True to run on a small subset)
    debug = False
