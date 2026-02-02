import os
import torch
import random
import numpy as np


class Config:
    """
    Central configuration for the Heterogeneous Stacking Network experiment.
    """

    # --- General ---
    seed = 42
    n_folds = 5
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Debugging
    debug = False  # Set to True to run on a small subset of data
    debug_sample_size = 100

    # --- Paths ---
    # Input data (Read-Only)
    input_dir = "./input"
    # Metadata contains the stratified train/val splits and the test set
    train_path = "./metadata/train.csv"
    val_path = "./metadata/val.csv"
    test_path = "./metadata/test.csv"
    sample_submission_path = "./input/sample_submission.csv"

    # Working directory for artifacts (Cache, Models)
    # Using 'idea_9' as the experiment identifier
    working_dir = "./working/idea_9"
    output_dir = os.path.join(working_dir, "output")
    model_dir = os.path.join(working_dir, "models")
    cache_dir = os.path.join(working_dir, "cache")
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # --- Model: Deep Semantic Branch (DeBERTa) ---
    model_name = "microsoft/deberta-v3-large"
    max_length = 1024
    dropout = 0.0  # Low dropout for regression fine-tuning

    # Training
    epochs = 4
    train_batch_size = 4  # Adjusted for A100 40GB with 1024 seq len
    valid_batch_size = 8
    gradient_accumulation_steps = 2
    max_grad_norm = 1000

    # Optimizer
    learning_rate = 1e-5
    weight_decay = 0.01
    eps = 1e-6
    betas = (0.9, 0.999)
    scheduler = "cosine"
    warmup_ratio = 0.1

    # AWP (Adversarial Weight Perturbation)
    use_awp = True
    awp_start_epoch = 2  # Start AWP after the model has stabilized
    awp_eps = 1e-4
    awp_lr = 1e-4

    # --- Model: Sparse Branches (Ridge) ---
    # Lexical (Word N-gram)
    word_ngram_range = (1, 3)
    word_min_df = 3

    # Morphological (Char N-gram)
    char_ngram_range = (3, 5)
    char_min_df = 3

    # --- Meta-Learner (LightGBM) ---
    lgbm_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "metric": "rmse",
        "random_state": seed,
        "verbose": -1,
        "n_jobs": -1,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    @classmethod
    def setup(cls):
        """
        Initializes the environment:
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        for d in [
            cls.working_dir,
            cls.output_dir,
            cls.model_dir,
            cls.cache_dir,
            cls.submission_dir,
        ]:
            os.makedirs(d, exist_ok=True)

        # Seed everything
        random.seed(cls.seed)
        os.environ["PYTHONHASHSEED"] = str(cls.seed)
        np.random.seed(cls.seed)
        torch.manual_seed(cls.seed)
        torch.cuda.manual_seed(cls.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Suppress tokenizer parallelism warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
