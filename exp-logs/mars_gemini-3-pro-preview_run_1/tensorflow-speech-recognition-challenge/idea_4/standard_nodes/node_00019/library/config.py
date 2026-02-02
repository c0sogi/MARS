import os


class Config:
    """
    Configuration class for Speech Command Recognition.
    Centralizes all hyperparameters, paths, and settings for the Dilated EfficientNet-B2 strategy.
    """

    # ==========================================
    # General Settings
    # ==========================================
    seed = 42
    num_workers = 12  # Utilizing available vCPUs
    device = "cuda"

    # ==========================================
    # Paths
    # ==========================================
    input_root = "./input"
    train_dir = os.path.join(input_root, "train", "audio")
    test_dir = os.path.join(input_root, "test", "audio")

    # Metadata paths (pre-generated)
    train_metadata_path = "./metadata/train.csv"
    val_metadata_path = "./metadata/val.csv"
    test_metadata_path = "./metadata/test.csv"

    # Working directory for caching and outputs
    # Specific to 'idea_4' as requested
    working_dir = "./working/idea_4"
    output_dir = working_dir

    # ==========================================
    # Audio Processing
    # ==========================================
    sample_rate = 16000
    duration = 1.0  # seconds
    n_samples = int(sample_rate * duration)

    # Mel Spectrogram parameters (High Fidelity)
    n_mels = 128
    n_fft = 1024  # 64ms window
    hop_length = 160  # 10ms step
    f_min = 0
    f_max = sample_rate // 2

    # ==========================================
    # Data Augmentation
    # ==========================================
    mixup_alpha = 1.0

    # SpecAugment parameters
    freq_mask_param = 20
    time_mask_param = 20

    # ==========================================
    # Model Architecture
    # ==========================================
    model_name = "efficientnet_b2"
    pretrained = True
    in_channels = 1

    # Structural modifications for Idea 4
    use_dilated_conv = (
        True  # Stride 1, Dilation 2 in final stage to preserve resolution
    )
    use_attentive_pooling = True  # Replaces Global Average Pooling

    # ==========================================
    # Labels
    # ==========================================
    # Target command labels
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

    # Full label set including auxiliary classes
    # Order is fixed to ensure consistent ID mapping across runs
    labels = target_labels + ["silence", "unknown"]
    num_classes = len(labels)

    # Mappings
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    epochs = 50
    batch_size = 64  # A100 40GB allows for larger batches
    learning_rate = 1e-3
    weight_decay = 1e-4
    min_lr = 1e-6

    # ==========================================
    # Runtime / Debugging
    # ==========================================
    # These can be overridden during initialization
    debug = False
    subset_size = None  # None means use full dataset

    def __init__(self, debug=False, subset_size=None, epochs=None):
        """
        Initialize config with optional overrides for debugging or quick runs.

        Args:
            debug (bool): If True, sets epochs to 2 and subset_size to 100 (if not specified).
            subset_size (int, optional): Limits the number of samples for training/validation.
            epochs (int, optional): Overrides the default number of epochs.
        """
        if debug:
            self.debug = True
            self.epochs = 2
            self.subset_size = 100 if subset_size is None else subset_size
        else:
            self.debug = False
            if subset_size is not None:
                self.subset_size = subset_size
            if epochs is not None:
                self.epochs = epochs

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

    @classmethod
    def get_label_map(cls):
        return cls.label2id

    @classmethod
    def get_id_map(cls):
        return cls.id2label
