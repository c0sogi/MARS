import os


class Config:
    # ==========================
    # General Configuration
    # ==========================
    seed = 42
    debug = False  # Set to True for fast debugging runs
    num_workers = 12  # Utilizing available vCPUs
    device = "cuda"

    # ==========================
    # Data Configuration
    # ==========================
    img_size = 384
    num_classes = 5

    # Input Paths
    input_root = "./input"
    metadata_dir = "./metadata"
    train_metadata = os.path.join(metadata_dir, "train.csv")
    val_metadata = os.path.join(metadata_dir, "val.csv")
    test_metadata = os.path.join(metadata_dir, "test.csv")

    # Image Directories
    train_images_dir = os.path.join(input_root, "train_images")
    test_images_dir = os.path.join(input_root, "test_images")

    # ==========================
    # Model Configuration
    # ==========================
    # Model A: Supervised Learning Expert (ViT)
    model_a_name = "vit_base_patch16_384"

    # Model B: Masked Image Modeling Expert (BEiT)
    model_b_name = "beit_base_patch16_384"

    # ==========================
    # Training Hyperparameters
    # ==========================
    epochs = 10
    # A100 40GB can handle larger batches, but 384x384 is memory intensive.
    # Using 32 to be safe and efficient with AMP.
    train_batch_size = 32
    valid_batch_size = 64

    # Optimizer (AdamW)
    lr = 2e-5
    min_lr = 1e-6
    weight_decay = 0.05

    # Regularization
    # Stochastic Depth (Drop Path) rate (Cite solution_lesson_node_00040)
    drop_path_rate = 0.1

    # Loss
    label_smoothing = 0.1

    # Scheduler (Cosine Annealing)
    # T_max corresponds to total epochs for a single cycle
    T_max = epochs

    # Early Stopping
    # Strategy explicitly requires disabling early stopping to allow full convergence
    # We set patience > epochs to effectively disable it while keeping the logic compatible
    patience = 20

    # ==========================
    # Output Configuration
    # ==========================
    working_dir = "./working/idea_9"
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Caching
    # Directory to store cached processed data if needed
    cache_dir = working_dir

    @classmethod
    def setup(cls):
        """Ensures necessary output directories exist."""
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)


# Initialize directories
Config.setup()
