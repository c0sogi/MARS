import os
import torch


class Config:
    """
    Centralized configuration for Author Identification Task.
    Implements parameters for Adversarily-Regularized Hybrid Stacking
    with Layer-Adaptive Fine-Tuning.
    """

    # =========================================================================
    # General & Reproducibility
    # =========================================================================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    # Input Metadata (Read-Only)
    train_path = "./metadata/train.csv"
    val_path = "./metadata/val.csv"
    test_path = "./metadata/test.csv"

    # Working Directory (Artifacts & Cache)
    output_dir = "./working/idea_9"
    os.makedirs(output_dir, exist_ok=True)

    # Submission Path
    submission_path = "./submission/submission.csv"
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Label Mapping
    label2id = {"EAP": 0, "HPL": 1, "MWS": 2}
    id2label = {0: "EAP", 1: "HPL", 2: "MWS"}
    num_classes = 3

    # Neural Input
    max_len = 85  # Optimized sequence length covering majority of distribution

    # Classical Features (TF-IDF + SVD)
    tfidf_word_ngram_range = (1, 3)
    tfidf_char_ngram_range = (2, 5)
    tfidf_min_df = 2
    svd_n_components = 100  # For XGBoost branch

    # =========================================================================
    # Model Architecture (Neural)
    # =========================================================================
    model_name = "microsoft/deberta-v3-large"
    dropout = 0.1

    # Custom Head Configuration
    n_last_hidden_layers = 4  # For Weighted Layer Pooling
    n_dropout_samples = 5  # For Multi-Sample Dropout

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    n_folds = 5
    epochs = 4

    # Batch Size Strategy (Target effective batch size >= 16)
    train_batch_size = 8
    valid_batch_size = 16
    gradient_accumulation_steps = 2

    # Optimizer
    lr = 1e-5  # Backbone learning rate
    head_lr = 1e-4  # Classifier head learning rate
    weight_decay = 0.01
    max_grad_norm = 1.0

    # Scheduler
    scheduler_type = "cosine"
    warmup_ratio = 0.1

    # Regularization / Early Stopping
    patience = 1

    # =========================================================================
    # Advanced Optimization Dynamics
    # =========================================================================
    # Layer-Wise Learning Rate Decay (LLRD)
    llrd_decay = 0.9

    # Adversarial Weight Perturbation (AWP)
    use_awp = True
    awp_start_epoch = 1  # Start AWP after the first epoch
    awp_eps = 1e-4  # Perturbation magnitude
    awp_lr = 1e-5  # AWP specific learning rate
