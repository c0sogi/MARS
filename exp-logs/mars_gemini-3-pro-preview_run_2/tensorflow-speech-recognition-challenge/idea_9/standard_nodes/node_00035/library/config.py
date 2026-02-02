import os
import torch


class Config:
    """
    Configuration class for the Speech Command Recognition task.
    Centralizes all hyperparameters and path definitions.
    """

    # ==========================================
    # General Settings
    # ==========================================
    seed = 42
    num_workers = 4  # Number of subprocesses for data loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # Data Paths
    # ==========================================
    input_root = "./input"
    train_metadata_path = "./metadata/train.csv"
    val_metadata_path = "./metadata/val.csv"
    test_metadata_path = "./metadata/test.csv"

    # Background noise folder for on-the-fly mixing
    background_noise_dir = os.path.join(
        input_root, "train", "audio", "_background_noise_"
    )

    # ==========================================
    # Output & Caching Paths
    # ==========================================
    # Directory for caching intermediate files and saving models
    working_dir = "./working/idea_9"

    # Ensure working directory exists
    os.makedirs(working_dir, exist_ok=True)

    # Model checkpoint path
    best_model_path = os.path.join(working_dir, "best_model.pth")

    # Submission file path
    submission_path = "./submission/submission.csv"
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # ==========================================
    # Audio Processing Parameters
    # ==========================================
    # High-Fidelity Spectral Oversampling Strategy
    sample_rate = 16000
    duration = 1.0  # seconds
    num_samples = int(sample_rate * duration)

    # STFT Parameters
    n_fft = 1024  # Larger FFT for spectral interpolation/oversampling
    hop_length = 160  # 10ms hop for high temporal resolution
    win_length = 400  # 25ms window

    # Mel Spectrogram Parameters
    n_mels = 128  # High resolution Mel bands
    f_min = 0
    f_max = 8000  # Nyquist frequency

    # ==========================================
    # Model Architecture
    # ==========================================
    backbone = "tf_efficientnetv2_b0"
    num_classes = 12
    in_channels = 1  # Spectrogram input is 1 channel
    pretrained = True

    # Multi-Head Attention Pooling
    attention_heads = 4

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    batch_size = 32
    epochs = 30
    learning_rate = 1e-3
    weight_decay = 1e-2

    # Optimization
    label_smoothing = 0.1

    # Scheduler (Cosine Annealing)
    T_max = epochs
    eta_min = 1e-6

    # Early Stopping
    early_stopping_patience = 5

    # ==========================================
    # Augmentation Parameters
    # ==========================================
    # SpecAugment
    freq_mask_param = 20  # Conservative masking
    time_mask_param = 30

    # Background Noise Injection
    noise_prob = 0.5
    min_snr_db = 10.0
    max_snr_db = 30.0

    # ==========================================
    # Debugging
    # ==========================================
    debug = False  # Set to True to run on a small subset for testing
    debug_sample_size = 200
