import os


class Config:
    # ==========================================
    # General Configuration
    # ==========================================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    debug_sample_size = 100
    num_workers = 8

    # ==========================================
    # Paths
    # ==========================================
    # Input Data
    input_root = "./input"
    train_metadata = "./metadata/train.csv"
    val_metadata = "./metadata/val.csv"
    test_metadata = "./metadata/test.csv"

    # Output Directories
    output_dir = "./working/idea_2"
    checkpoint_path = os.path.join(output_dir, "best_model.pth")

    # Submission
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # ==========================================
    # Audio Parameters
    # ==========================================
    sample_rate = 32000
    duration = 30  # seconds
    n_mels = 128
    n_fft = 1024
    hop_length = 320
    fmin = 0
    fmax = None  # None implies sample_rate // 2

    # ==========================================
    # Model Architecture
    # ==========================================
    backbone = "efficientnet_b0"
    pretrained = True
    num_classes = 80
    in_channels = 1  # Spectrograms are 1 channel
    use_attention_pooling = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    batch_size = 32
    epochs = 20
    learning_rate = 1e-3
    weight_decay = 1e-4

    # Scheduler (OneCycleLR)
    pct_start = 0.3
    div_factor = 25
    final_div_factor = 1000

    # Augmentation
    mixup_alpha = 0.2
    spec_augment_time_mask = 30
    spec_augment_freq_mask = 20

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.output_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)


# Initialize directories immediately
Config.setup()
