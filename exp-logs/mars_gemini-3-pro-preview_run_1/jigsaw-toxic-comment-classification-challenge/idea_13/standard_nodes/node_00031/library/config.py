import os
import torch


class Config:
    # ====================================================
    # General Settings
    # ====================================================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    n_folds = 5
    num_workers = 4
    print_freq = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====================================================
    # Data Settings
    # ====================================================
    max_len = 512
    train_batch_size = 16
    valid_batch_size = 32

    # Target Columns
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
    # Model Architecture Settings
    # ====================================================
    model_name = "microsoft/deberta-v3-base"

    # Deep Supervision
    use_deep_supervision = True
    deep_supervision_layer = (
        6  # Attach aux head to the 6th layer (middle of base model)
    )
    aux_loss_weight = 0.3

    # Main Head Aggregation
    num_last_layers_agg = 4  # Weighted aggregation of the last 4 layers

    # Dropout
    fc_dropout = 0.2

    # ====================================================
    # Training Settings
    # ====================================================
    epochs = 5

    # Optimization
    lr = 2e-5
    min_lr = 1e-6
    weight_decay = 0.01
    eps = 1e-6
    max_grad_norm = 1000

    # Scheduler
    scheduler_type = "OneCycleLR"  # Matches the idea description
    pct_start = 0.1  # Warmup percentage for OneCycle

    # Adversarial Weight Perturbation (AWP)
    use_awp = True
    awp_start_epoch = 2  # Start AWP from epoch 2
    awp_eps = 1e-4
    awp_lr = 1e-4

    # ====================================================
    # Path Settings
    # ====================================================
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Working directory for this specific idea
    working_dir = "./working/idea_13"
    output_dir = os.path.join(working_dir, "output")

    # Cache paths
    train_cache_path = os.path.join(working_dir, "train_cache.parquet")
    test_cache_path = os.path.join(working_dir, "test_cache.parquet")
    val_cache_path = os.path.join(working_dir, "val_cache.parquet")

    # Model save path
    model_save_path = os.path.join(working_dir, "model.pth")
    submission_path = os.path.join(output_dir, "submission.csv")

    @classmethod
    def setup(cls):
        """
        Creates necessary directories.
        """
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)


# Initialize directories immediately upon import
Config.setup()
