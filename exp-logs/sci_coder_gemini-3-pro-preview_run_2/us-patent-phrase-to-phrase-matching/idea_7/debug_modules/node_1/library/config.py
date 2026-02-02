import os
import torch


class Config:
    """
    Configuration class for the Heterogeneous Adversarial Hybrid Stacking solution.
    """

    # ====================================================
    # General Settings
    # ====================================================
    seed = 42
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug = False  # Set to True to train on a small subset for debugging

    # ====================================================
    # Paths
    # ====================================================
    # Input paths (using generated metadata)
    metadata_dir = "./metadata"
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")

    # Output and Working paths
    working_dir = "./working/idea_7"
    output_dir = os.path.join(working_dir, "models")
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Create directories
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # ====================================================
    # Model Architecture
    # ====================================================
    # Heterogeneous Ensemble: Two different backbones
    models = [
        {
            "model_name": "microsoft/deberta-v3-large",
            "tokenizer_name": "microsoft/deberta-v3-large",
            "short_name": "deberta_v3_large",
        },
        {
            "model_name": "roberta-large",
            "tokenizer_name": "roberta-large",
            "short_name": "roberta_large",
        },
    ]

    max_length = 133
    num_classes = 5  # Classification: 0.0, 0.25, 0.5, 0.75, 1.0

    # Feature Flags
    use_structural_features = True
    structural_features = ["norm_levenshtein", "jaccard", "len_ratio"]
    enrich_context = True  # Map CPC codes to full text descriptions

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    n_folds = 5
    epochs = 5

    # Effective Batch Size = batch_size * gradient_accumulation_steps
    # Target: 32
    batch_size = 16
    gradient_accumulation_steps = 2

    # Optimizer & Scheduler
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    min_lr = 1e-6
    weight_decay = 0.01
    max_grad_norm = 1.0
    label_smoothing = 0.1

    scheduler_type = "cosine"
    warmup_ratio = 0.1

    # ====================================================
    # Adversarial Weight Perturbation (AWP)
    # ====================================================
    use_awp = True
    awp_start_epoch = 2  # Start AWP after this epoch (1-based count, so after epoch 1)
    awp_eps = 1e-4
    awp_lr = 1e-4

    # ====================================================
    # Logging & Validation
    # ====================================================
    print_freq = 50
    val_check_interval = 1.0  # Check validation at the end of epoch
