import os
import torch


class CFG:
    # General Config
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Paths
    input_root = "./input"
    train_csv = "./metadata/train.csv"
    val_csv = "./metadata/val.csv"
    test_csv = "./metadata/test.csv"

    # Output Directory
    # Ensure this directory exists for saving models and cache
    output_dir = "./working/idea_2"
    os.makedirs(output_dir, exist_ok=True)

    # Audio Processing Config
    sample_rate = 32000  # Resample audio to 32kHz
    duration = 5  # Duration of audio crops for training in seconds
    n_fft = 1024  # FFT window size
    hop_length = 320  # Hop length for STFT (10ms at 32kHz)
    n_mels = 128  # Number of Mel bands (Required: 128)
    fmin = 20  # Minimum frequency
    fmax = 16000  # Maximum frequency (Nyquist at 32kHz)

    # Model Config
    model_name = "efficientnet_b2"
    pretrained = True
    num_classes = 80
    in_channels = 1  # Spectrogram input channel

    # Training Config
    epochs = 25
    batch_size = 64  # Adjust based on GPU memory (A100 40GB is large)
    lr = 1e-3
    min_lr = 1e-6
    weight_decay = 1e-2

    # Augmentation Config
    mixup_alpha = 1.0  # Probability/Strength for Mixup
    mixup_prob = 0.5  # Probability of applying mixup
    spec_aug_time_mask = 30  # Max time mask width
    spec_aug_freq_mask = 20  # Max freq mask width

    # Inference Config
    inference_batch_size = 32
