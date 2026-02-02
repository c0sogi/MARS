import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input data paths
    # We use the full original train file for 5-fold CV to maximize data usage
    raw_train_path = "./input/train.csv"
    raw_test_path = "./input/test.csv"
    sample_submission_path = "./input/sample_submission_null.csv"

    # Metadata paths (if needed for specific validation splits)
    metadata_train_path = "./metadata/train.csv"
    metadata_val_path = "./metadata/validation.csv"

    # Output directories
    working_dir = "./working/idea_5/"
    output_dir = "./working/idea_5/"
    submission_path = "./submission/submission.csv"

    # Ensure working directory exists
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    model_name = "microsoft/deberta-v3-base"
    max_len = 128
    hidden_size = 768  # Hidden size for DeBERTa-v3-base
    dropout = 0.1
    num_classes = 1

    # Weighted Layer Pooling Settings
    # We use the last N layers for pooling
    pool_layers = 4

    # =========================================================================
    # Structural Feature Engineering
    # =========================================================================
    # TF-IDF Settings
    tfidf_word_ngram_range = (1, 2)
    tfidf_char_ngram_range = (3, 5)

    # Dimensionality Reduction
    svd_components = 256

    # Fusion Settings
    # Dimension to project SVD features to before fusion
    fusion_dim = 768

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    n_folds = 5
    epochs = 5
    batch_size = 16
    gradient_accumulation_steps = 1

    # Differential Learning Rates
    lr_backbone = 2e-5  # Lower rate for pre-trained layers
    lr_head = 1e-3  # Higher rate for new layers (Pooling, Fusion, Classifier)

    # Optimization
    weight_decay = 0.01
    scheduler_type = "cosine"  # or 'linear'
    warmup_ratio = 0.05
    max_grad_norm = 1.0

    # Adversarial Weight Perturbation (AWP)
    use_awp = True
    awp_lr = 1e-4
    awp_eps = 1e-4
    awp_start_epoch = 1  # Start AWP after the first epoch

    # Early Stopping
    patience = 3

    # =========================================================================
    # Caching
    # =========================================================================
    # Filenames for cached features
    cache_train_features = os.path.join(working_dir, "train_features.parquet")
    cache_test_features = os.path.join(working_dir, "test_features.parquet")
    cache_svd_model = os.path.join(
        working_dir, "svd_model.joblib"
    )  # If using joblib, or pickle
    # We prefer .npy for numpy arrays
    cache_train_svd = os.path.join(working_dir, "train_svd.npy")
    cache_test_svd = os.path.join(working_dir, "test_svd.npy")
