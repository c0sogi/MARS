import os
import torch


class Config:
    """
    Configuration class for the Robust Hybrid DeBERTa-v3 experiment.
    Includes hyperparameters for model, data, training, and optimization.
    """

    # ====================================================
    # General Settings
    # ====================================================
    seed = 42
    debug = False  # Set to True to run on a small subset of data for debugging
    num_workers = 2  # Number of dataloader workers

    # ====================================================
    # Data Paths
    # ====================================================
    # Using metadata directory as source of truth for splits
    metadata_dir = "./metadata"
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "validation.csv")
    test_path = os.path.join(metadata_dir, "test.csv")

    # Output Directories
    working_dir = "./working/idea_4/"
    submission_dir = "./submission"
    submission_file = os.path.join(submission_dir, "submission.csv")

    # ====================================================
    # Model Architecture
    # ====================================================
    model_name = "microsoft/deberta-v3-base"
    max_len = 128  # Max sequence length for tokenizer
    svd_components = 256  # Dimension of the structural feature vector (TF-IDF -> SVD)
    hidden_size = 768  # Hidden size of the backbone model

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    n_folds = 5
    epochs = 5
    batch_size = 16

    # ====================================================
    # Optimization
    # ====================================================
    # Differential Learning Rates
    lr_backbone = 2e-5  # Lower LR for pre-trained weights
    lr_head = 1e-3  # Higher LR for the new classification head

    weight_decay = 0.01
    warmup_ratio = 0.05
    scheduler_type = "linear"

    # ====================================================
    # Robustness & Regularization
    # ====================================================
    # Multi-Sample Dropout: Use multiple dropout masks to improve generalization
    dropout_rates = [0.1, 0.15, 0.2, 0.25, 0.3]

    # Adversarial Weight Perturbation (AWP)
    use_awp = True
    awp_start_epoch = 2  # Start AWP after the model has stabilized
    awp_lr = 1e-4  # Learning rate for the adversarial attack
    awp_eps = 1e-2  # Epsilon for weight perturbation

    # ====================================================
    # Hardware
    # ====================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)

        # Suppress tokenizers parallelism warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Execute setup immediately
Config.setup()
