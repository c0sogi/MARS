import os
import torch


class Config:
    # ==========================================
    # System Settings
    # ==========================================
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 12  # Using available vCPUs

    # ==========================================
    # File Paths
    # ==========================================
    # Input Data (Read-Only)
    input_dir = "./input"

    # Metadata (Pre-generated)
    metadata_dir = "./metadata"
    train_metadata_path = os.path.join(metadata_dir, "train.csv")
    val_metadata_path = os.path.join(metadata_dir, "val.csv")
    test_metadata_path = os.path.join(metadata_dir, "test.csv")

    # Working Directory (For caching, checkpoints, logs)
    # Using idea_16 to isolate this specific solution iteration
    working_dir = "./working/idea_16"
    os.makedirs(working_dir, exist_ok=True)

    # Submission Output
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    # ==========================================
    # Data Parameters
    # ==========================================
    image_size = 224  # Fixed resolution to avoid overfitting/instability
    num_classes = 120
    batch_size = 64  # Optimized for A100 40GB with ConvNeXt-Small

    # ==========================================
    # Model Architecture
    # ==========================================
    # ConvNeXt-Small pretrained on ImageNet-21k (in12k) and finetuned on 1k
    # This provides the rich feature priors required for the task
    model_name = "convnext_small.in12k_ft_in1k"

    # ==========================================
    # Training Strategy (Corrected Transfer Learning)
    # ==========================================
    n_folds = 5

    # Phase 1: Head Warmup (Frozen Backbone)
    # High LR to align random head with pretrained features
    warmup_epochs = 1
    warmup_lr = 1e-3

    # Phase 2: Full Fine-tuning
    # Conservative LR to preserve pretrained features
    finetune_epochs = 30
    finetune_lr = 1e-5
    min_lr = 1e-7  # Floor for Cosine Annealing
    weight_decay = 1e-4

    # ==========================================
    # Ensembling (Manual Model Soup)
    # ==========================================
    # Number of best epochs (by Val Loss) to average weights for
    soup_top_k = 3

    # ==========================================
    # Debugging
    # ==========================================
    debug = False
    debug_sample_size = 200  # Subset size if debug is True
