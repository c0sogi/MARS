import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    debug = False  # Set to True to run on a small subset for testing
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Paths
    # =========================================================================
    # Read-only input directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea (Idea 3)
    WORKING_DIR = "./working/idea_3"

    # Sub-directories for caching and models
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Metadata file paths
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_name = "microsoft/deberta-v3-large"
    num_labels = 1  # Regression output

    # Input handling
    max_length = 512

    # Sliding Window Parameters for Inference/Embedding Extraction
    chunk_size = 512
    stride = 128

    # Pooling Strategy
    pooling = "attention"  # Options: 'mean', 'attention', 'cls'

    # =========================================================================
    # Training Hyperparameters (Stage 1: Backbone Fine-tuning)
    # =========================================================================
    epochs = 4
    train_batch_size = 4  # Lower batch size for Large model on 40GB
    valid_batch_size = 8
    gradient_accumulation_steps = 2  # Effective batch size = 4 * 2 = 8

    # Mixed Precision
    use_amp = True

    # Optimization
    weight_decay = 0.01
    max_grad_norm = 10.0
    scheduler_type = "cosine"
    num_warmup_steps_ratio = 0.1

    # Layer-wise Learning Rate Decay (LLRD)
    # Higher LR for the task-specific head, lower for the backbone
    head_lr = 1e-4
    backbone_lr = 1e-5
    llrd_decay = 0.9  # Multiplicative decay factor for lower layers

    # =========================================================================
    # LightGBM Hyperparameters (Stage 2: Stacking)
    # =========================================================================
    lgbm_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": 0.01,
        "num_leaves": 31,
        "feature_fraction": 0.7,  # Regularization: use 70% of features per tree
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "random_state": seed,
        "n_jobs": -1,
    }
