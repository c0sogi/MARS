import os
import torch


class AudioConfig:
    """
    Configuration for Audio Processing.
    Targeting high-resolution Log-Mel Spectrograms for ConvNeXt.
    """

    sr = 2000  # Sampling rate from analysis
    duration = 2.0  # Fixed duration in seconds
    n_mels = 320  # High vertical resolution as per Idea
    fmin = 0
    fmax = 1000  # Nyquist frequency for 2000Hz SR
    n_fft = 1024  # Sufficient frequency resolution
    hop_length = 20  # ~10ms at 2000Hz (preserves temporal dynamics)
    win_length = 100  # ~50ms window

    # Derived parameters
    num_samples = int(sr * duration)


class ModelConfig:
    """
    Configuration for the Neural Network Architecture.
    Uses ConvNeXt-Small backbone with custom Attention and Pooling.
    """

    model_name = "convnext_small.fb_in1k"
    pretrained = True
    in_chans = 1
    num_classes = 1

    # Structural components
    use_gem = True  # Generalized Mean Pooling
    use_coord_att = True  # Coordinate Attention

    # Regularization
    drop_rate = 0.3  # Head dropout
    drop_path_rate = 0.2  # Stochastic depth rate


class TrainConfig:
    """
    Configuration for Training, Optimization, and Paths.
    """

    # Reproducibility
    seed = 42

    # Compute
    batch_size = 32
    num_workers = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optimization
    epochs = 20
    lr = 3e-4
    min_lr = 1e-6
    weight_decay = 0.05

    # Augmentation
    mixup_alpha = 0.4
    spec_aug_time_mask = 30
    spec_aug_freq_mask = 30

    # Debugging / Development flags
    debug = False  # Set to True to train on a small subset
    debug_samples = 200  # Number of samples to use in debug mode

    # Paths - Input
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Paths - Output / Cache
    # Ensure directories exist
    CACHE_DIR = "./working/idea_7"
    os.makedirs(CACHE_DIR, exist_ok=True)

    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    CHECKPOINT_PATH = os.path.join(CACHE_DIR, "best_model.pth")
