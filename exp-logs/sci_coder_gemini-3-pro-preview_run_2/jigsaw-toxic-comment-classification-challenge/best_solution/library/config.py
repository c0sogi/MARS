import os
import torch


class Config:
    """
    Centralized configuration for the Toxicity Prediction pipeline.
    Handles file paths, model hyperparameters, training settings, and hardware configuration.
    """

    # ====================================================
    # General Settings
    # ====================================================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    num_workers = 4

    # ====================================================
    # Directory & File Paths
    # ====================================================
    # Input directories (Read-Only)
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Working directory (Read/Write) - for caching features and models
    working_dir = "./working/idea_3"

    # Specific file paths
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Output path for final submission
    submission_path = "./submission/submission.csv"

    # Create necessary directories
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # ====================================================
    # Data Information
    # ====================================================
    target_cols = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    num_classes = len(target_cols)

    # ====================================================
    # Model Architecture
    # ====================================================
    # Branch A: DeBERTa-v3
    model_1_name = "microsoft/deberta-v3-base"

    # Branch B: RoBERTa
    model_2_name = "roberta-base"

    # Multi-Sample Dropout settings (used in Transformer heads)
    # Using multiple dropout masks to improve generalization
    msd_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    fc_dropout = 0.0  # Dropout before the MSD layer (usually 0 if MSD is used)

    # ====================================================
    # Tokenizer & Input
    # ====================================================
    max_len = 200  # Extended context window as per strategy
    tokenizer_names = [model_1_name, model_2_name]

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    epochs = 4
    train_batch_size = 32  # Increased for A100 to speed up training
    valid_batch_size = 64

    # Optimization
    lr = 2e-5  # Base learning rate
    head_lr = 1e-4  # Higher learning rate for the classification head
    min_lr = 1e-6  # Minimum learning rate for scheduler
    weight_decay = 0.01
    max_grad_norm = 10.0

    # Layer-wise Learning Rate Decay (LLRD)
    llrd_decay = 0.9  # Decay factor for deeper layers

    # Scheduler
    scheduler = "cosine"  # Cosine Annealing with Warmup
    warmup_ratio = 0.1  # Percentage of steps for warmup

    # Early Stopping
    patience = 3  # Stop after 3 epochs of no improvement in validation AUC

    # ====================================================
    # Linear Model (Branch C)
    # ====================================================
    # TF-IDF settings
    tfidf_word_ngram_range = (1, 2)
    tfidf_char_ngram_range = (2, 6)
    tfidf_max_features_word = 150000
    tfidf_max_features_char = 50000

    # ====================================================
    # Hardware
    # ====================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====================================================
    # Ensemble Weights (Initial Guess)
    # ====================================================
    # These can be optimized via validation search
    # Order: [DeBERTa, RoBERTa, Linear]
    initial_weights = [0.4, 0.4, 0.2]

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 30)
        print("CONFIG")
        print("=" * 30)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k}: {v}")
        print("=" * 30)
