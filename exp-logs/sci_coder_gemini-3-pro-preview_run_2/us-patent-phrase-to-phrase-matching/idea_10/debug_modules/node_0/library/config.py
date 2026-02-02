import os
import torch
import random
import numpy as np


class Config:
    # =========================================================================
    # General Configuration
    # =========================================================================
    seed = 42
    debug = False  # Set to True to run with a small subset of data for debugging
    num_workers = 4

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_10"

    # Create necessary working directories
    os.makedirs(working_dir, exist_ok=True)

    model_output_dir = os.path.join(working_dir, "models")
    os.makedirs(model_output_dir, exist_ok=True)

    cache_dir = os.path.join(working_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Data Paths (using metadata splits)
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Path for caching processed CPC text descriptions
    cpc_text_path = os.path.join(working_dir, "cpc_texts.parquet")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_name = "microsoft/deberta-v3-large"
    max_length = 133  # Optimized based on text length analysis
    dropout = 0.0  # Low dropout for DeBERTa

    # 5 discrete score classes: {0.0, 0.25, 0.5, 0.75, 1.0}
    num_classes = 5

    # Structural features to be fused in the model
    structural_features = [
        "normalized_levenshtein",
        "jaccard_similarity",
        "length_ratio",
    ]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    n_folds = 5
    epochs = 5

    # Batch Size Strategy for 24GB+ VRAM
    train_batch_size = 8
    valid_batch_size = 16
    gradient_accumulation_steps = 4  # Effective Batch Size = 8 * 4 = 32

    # Optimization
    learning_rate = 2e-5
    weight_decay = 0.01
    max_grad_norm = 1.0
    scheduler = "cosine"
    warmup_ratio = 0.1

    # =========================================================================
    # Advanced Training Techniques
    # =========================================================================
    # Adversarial Weight Perturbation (AWP)
    use_awp = True
    awp_start_epoch = 1  # Start AWP after the 1st epoch (0-indexed)
    awp_eps = 1e-4
    awp_lr = 1e-4

    # Gaussian Soft Targets
    # Sigma controls the spread of probability mass to neighboring classes
    label_sigma = 1.0

    # =========================================================================
    # Meta-Learner (Stacking) Configuration
    # =========================================================================
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 2000,
        "early_stopping_rounds": 100,
        "verbose": -1,
        "random_state": seed,
        "n_jobs": -1,
    }

    # =========================================================================
    # Hardware
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def set_seed(cls):
        """Sets the random seed for reproducibility."""
        random.seed(cls.seed)
        np.random.seed(cls.seed)
        torch.manual_seed(cls.seed)
        torch.cuda.manual_seed_all(cls.seed)
        os.environ["PYTHONHASHSEED"] = str(cls.seed)


# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
