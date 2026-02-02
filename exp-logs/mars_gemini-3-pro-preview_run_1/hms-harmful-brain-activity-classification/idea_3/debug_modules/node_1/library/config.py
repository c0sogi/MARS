import os
import torch


class Config:
    """
    Configuration class for the Context-Aware Dual-Stream Hybrid Network.
    Centralizes all file paths, hyperparameters, and global settings.
    """

    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    seed = 42
    debug = False  # If True, runs on a small subset of data
    debug_subset_size = 500  # Number of samples to use in debug mode
    num_workers = 4  # Number of dataloader workers

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_3"
    submission_dir = "./submission"

    # Data Directories
    train_eeg_dir = os.path.join(input_dir, "train_eegs")
    train_spec_dir = os.path.join(input_dir, "train_spectrograms")
    test_eeg_dir = os.path.join(input_dir, "test_eegs")
    test_spec_dir = os.path.join(input_dir, "test_spectrograms")

    # Metadata Files
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "val.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Output Files
    submission_path = os.path.join(submission_dir, "submission.csv")
    model_save_path = os.path.join(working_dir, "best_model.pth")

    # -------------------------------------------------------------------------
    # Data Parameters
    # -------------------------------------------------------------------------
    # Target Columns (Probabilities used for training)
    prob_cols = [
        "seizure_prob",
        "lpd_prob",
        "gpd_prob",
        "lrda_prob",
        "grda_prob",
        "other_prob",
    ]

    # Vote Columns (Original counts, mostly for reference)
    vote_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    class_names = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
    num_classes = 6

    # EEG Signal Parameters
    eeg_sr_original = 200  # Original Sampling Rate (Hz)
    eeg_sr_target = 100  # Downsampled Rate (Hz)
    eeg_duration = 50  # Duration in seconds
    eeg_seq_len = int(eeg_duration * eeg_sr_target)  # 5000 samples
    eeg_channels = 20  # 19 EEG + 1 EKG

    # Ordered list of channels expected in the parquet files
    channel_names = [
        "Fp1",
        "F3",
        "C3",
        "P3",
        "F7",
        "T3",
        "T5",
        "O1",
        "Fz",
        "Cz",
        "Pz",
        "Fp2",
        "F4",
        "C4",
        "P4",
        "F8",
        "T4",
        "T6",
        "O2",
        "EKG",
    ]

    # Spectrogram Parameters
    spec_size = (512, 512)  # Input size for 2D CNN (Height, Width)

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    # 2D Stream (Spectrograms)
    backbone_2d = "tf_efficientnet_b0.ns_jft_in1k"
    pretrained_2d = True

    # 1D Stream (Raw EEG) - Multi-Scale CNN
    kernel_sizes_1d = [3, 5, 7]
    filters_1d = [32, 64, 128]

    # Fusion Head
    drop_rate = 0.5  # Dropout before final classification
    fusion_hidden_dim = (
        128  # Dimension to project both streams to before concatenation (optional)
    )

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    batch_size = 32
    epochs = 10
    lr = 1e-3
    weight_decay = 0.01
    max_grad_norm = 10.0
    patience = 3  # Early stopping patience

    # Scheduler
    pct_start = 0.3  # Percentage of training to increase LR (OneCycle)
    div_factor = 25  # Initial LR = max_lr / div_factor
    final_div_factor = 1e4  # Final LR = initial_lr / final_div_factor

    # Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = True  # Automatic Mixed Precision

    def __init__(self, **kwargs):
        """
        Initialize Config with optional overrides.
        Creates necessary directories.
        """
        # Update attributes with passed kwargs
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # Create directories
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

    def __repr__(self):
        """Pretty print configuration."""
        attrs = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
        # Also include class attributes that weren't overridden
        for k in dir(self):
            if not k.startswith("__") and not callable(getattr(self, k)):
                if k not in attrs:
                    attrs[k] = getattr(self, k)

        return f"Config({attrs})"
