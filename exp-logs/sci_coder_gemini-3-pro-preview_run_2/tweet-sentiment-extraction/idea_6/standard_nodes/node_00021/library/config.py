import os
import torch


class Config:
    """
    Configuration class for the Tweet Sentiment Extraction task (Idea 6).
    Encapsulates model hyperparameters, training settings, and file paths.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    n_folds = 5
    debug = False  # Set to True to run on a small subset for debugging
    debug_sample_size = 500  # Number of samples to use when debug=True
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Metadata (Generated in ./metadata as per instructions)
    TRAIN_META = "./metadata/train.csv"
    VAL_META = "./metadata/val.csv"
    TEST_META = "./metadata/test.csv"
    SAMPLE_SUBMISSION = "./input/sample_submission.csv"

    # Working Directory for Idea 6
    # Stores cache, models, and logs
    WORKING_DIR = "./working/idea_6/"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_OUTPUT_DIR = os.path.join(WORKING_DIR, "models")

    # Final Submission Path
    SUBMISSION_FILE = "./submission/submission.csv"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_name = "microsoft/deberta-v3-large"
    max_len = 128

    # Multi-Sample Dropout Settings
    # Dropout rates for the parallel branches in the prediction head
    msd_dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    epochs = 5
    train_batch_size = 8  # Conservative size for DeBERTa-Large on A100
    valid_batch_size = 16

    # Optimizer Settings
    learning_rate = 2e-5  # Uniform learning rate
    weight_decay = 0.01
    eps = 1e-6
    max_grad_norm = 1.0

    # Scheduler Settings
    scheduler_type = "cosine"
    warmup_ratio = 0.1

    # Loss Function Settings
    label_smoothing = 0.1
    jaccard_weight = 0.5  # Weight for Soft Jaccard Loss component

    # =========================================================================
    # Inference Settings
    # =========================================================================
    # Heuristic: If sentiment is 'neutral', predict the entire text.
    # This exploits the high correlation (Jaccard ~0.98) for neutral tweets.
    neutral_heuristic = True
