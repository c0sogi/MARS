import os
import torch


class CFG:
    """
    Configuration class for the Adversarial Hybrid DeBERTa-v3-Large Ensemble.
    """

    # ====================================================
    # General Settings
    # ====================================================
    debug = False
    num_workers = 4
    seed = 42
    n_fold = 5

    # ====================================================
    # Paths
    # ====================================================
    # Metadata paths (Input)
    train_file = "./metadata/train.csv"
    val_file = "./metadata/val.csv"
    test_file = "./metadata/test.csv"

    # Output paths
    working_dir = "./working/idea_6/"
    output_dir = os.path.join(working_dir, "models")
    submission_dir = "./submission/"
    submission_file = os.path.join(submission_dir, "submission.csv")

    # ====================================================
    # Model Architecture
    # ====================================================
    model_name = "microsoft/deberta-v3-large"
    max_len = 140
    target_size = 5  # 5 classes: 0.0, 0.25, 0.5, 0.75, 1.0

    # Architecture specifics
    dropout = 0.1
    fc_dropout = 0.1

    # ====================================================
    # Training Parameters
    # ====================================================
    epochs = 5
    train_batch_size = 8
    valid_batch_size = 16
    gradient_accumulation_steps = 4  # Effective batch size = 32
    max_grad_norm = 1.0

    # ====================================================
    # Optimization
    # ====================================================
    encoder_lr = 1e-5
    decoder_lr = 2e-5
    min_lr = 1e-7
    weight_decay = 0.01
    eps = 1e-6
    betas = (0.9, 0.999)

    # Scheduler
    scheduler_type = "cosine"
    num_warmup_steps = 0
    batch_scheduler = True

    # ====================================================
    # Adversarial Training (AWP)
    # ====================================================
    awp = True
    awp_lr = 1e-4
    awp_eps = 1e-2
    awp_start_epoch = 1  # Start AWP after epoch 0 (1st epoch)

    # ====================================================
    # Environment
    # ====================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_freq = 100

    @classmethod
    def setup(cls, debug=None):
        """
        Initialize directories and override settings for debugging.

        Args:
            debug (bool, optional): If True, overrides the debug flag to True.
        """
        if debug is not None:
            cls.debug = debug

        if cls.debug:
            print("DEBUG Mode Activated: Reducing epochs and data size.")
            cls.epochs = 2
            cls.n_fold = 2
            cls.print_freq = 10

        # Create directories
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)

        print(f"Configuration Setup Complete. Device: {cls.device}")
        print(f"Output Dir: {cls.output_dir}")
