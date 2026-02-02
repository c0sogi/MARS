import os
import torch


class Config:
    """
    Centralized configuration for the Siamese DeBERTa-v3-Base pipeline.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    exp_name = "idea_5"

    # =========================================================================
    # Paths
    # =========================================================================
    # Input Metadata (Read-Only)
    train_path = "./metadata/train_metadata.csv"
    val_path = "./metadata/val_metadata.csv"
    test_path = "./metadata/test_metadata.csv"

    # Output Directories
    working_dir = f"./working/{exp_name}/"
    cache_dir = os.path.join(working_dir, "cache")
    model_save_path = os.path.join(working_dir, "best_model.pth")

    # Submission
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_name = "microsoft/deberta-v3-base"
    max_length = 512

    # Weighted Layer Pooling Settings
    n_last_layers = 4  # Number of last hidden states to use for weighted pooling

    # Dropout
    hidden_dropout_prob = 0.1
    attention_probs_dropout_prob = 0.1

    # Feature Engineering
    # Features: (Char Diff, Char Ratio, Word Diff, Word Ratio, Newline Diff, Newline Ratio)
    num_scalar_features = 6

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    epochs = 3

    # Batch Sizes
    # Physical batch size is kept small to fit in GPU memory with max_len=512
    train_batch_size = 4
    valid_batch_size = 8

    # Gradient Accumulation to simulate larger effective batch size
    # Effective Batch Size = train_batch_size * gradient_accumulation_steps
    # 4 * 4 = 16
    gradient_accumulation_steps = 4

    # Optimization
    # Differential Learning Rates
    lr_backbone = 1e-5  # Slower learning for the pre-trained transformer
    lr_head = 1e-3  # Faster learning for the custom head and pooling weights

    weight_decay = 0.01
    max_grad_norm = 1.0

    # Early Stopping
    patience = 2  # Stop if validation loss doesn't improve for 2 evaluations

    # =========================================================================
    # Environment & Hardware
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4
    use_fp16 = True  # Use Mixed Precision training

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        """
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.cache_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)

        # Set environment variable for tokenizers parallelism to avoid warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
