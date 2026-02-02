import os
import torch


class CFG:
    """
    Configuration class for the Phrase Similarity Task.
    Includes model hyperparameters, training settings, AWP configuration,
    and file paths.
    """

    # ==============================
    # General Configuration
    # ==============================
    debug = False  # Set to True for fast debugging on a subset of data
    num_workers = 4  # Number of dataloader workers
    print_freq = 100  # Logging frequency in steps
    seed = 42  # Random seed for reproducibility

    # ==============================
    # Model Architecture
    # ==============================
    model_name = "microsoft/deberta-v3-large"
    max_len = 140  # Max sequence length (Context + Anchor + Target)
    target_size = 1  # Regression output
    fc_dropout = 0.2  # Dropout rate for the final fully connected layer

    # ==============================
    # Training Hyperparameters
    # ==============================
    epochs = 4  # Total number of training epochs
    batch_size = 16  # Batch size per GPU (A100 can handle 16-32 for Large)
    gradient_accumulation_steps = 1
    max_grad_norm = 1000  # Gradient clipping

    # Optimizer & Scheduler
    encoder_lr = 2e-5  # Learning rate for the transformer backbone
    decoder_lr = 2e-5  # Learning rate for the head
    min_lr = 1e-6  # Minimum learning rate for scheduler
    eps = 1e-6  # Optimizer epsilon
    betas = (0.9, 0.999)  # AdamW betas
    weight_decay = 0.01  # Weight decay
    scheduler = "cosine"  # Learning rate scheduler type
    num_warmup_steps = 0  # Warmup steps
    batch_scheduler = True  # Step scheduler every batch
    num_cycles = 0.5  # Cosine cycles

    # ==============================
    # Adversarial Weight Perturbation (AWP)
    # ==============================
    awp = True  # Enable AWP
    awp_lr = 1e-4  # Learning rate for adversarial perturbation
    awp_eps = 1e-4  # Epsilon for AWP
    awp_start_epoch = (
        1  # Start AWP after this many epochs (0-indexed, so 1 means start at epoch 2)
    )

    # ==============================
    # Cross-Validation
    # ==============================
    n_fold = 5  # Number of folds
    trn_fold = [0, 1, 2, 3, 4]  # Folds to train

    # ==============================
    # Paths & Directories
    # ==============================
    # Input Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Specific File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    CPC_DATA = os.path.join(INPUT_DIR, "description.md")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Directories
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"
    OUTPUT_DIR = WORKING_DIR  # Alias for model saving

    # ==============================
    # Hardware
    # ==============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==============================
    # Setup Logic
    # ==============================
    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Execute setup immediately upon import to ensure directories exist
CFG.setup()
