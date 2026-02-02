import os
import torch


class CFG:
    """
    Configuration module for Apple Disease Detection.
    Strategy: Decoupled Calibration with Full-Data Seed Ensembling.

    Phase 1: Proxy Calibration (5-Fold CV) to determine optimal epoch (E_opt).
    Phase 2: Production Training (Seed Ensemble) on 100% data using E_opt.
    """

    # ==========================================
    # General Configuration
    # ==========================================
    project_name = "apple_disease_detection"
    idea_name = "idea_15"
    debug = False
    seed = 42
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # Directory Setup
    # ==========================================
    # Input Directories (Read-Only)
    input_dir = "./input"
    images_dir = os.path.join(input_dir, "images")

    # Metadata Directories (Pre-generated)
    metadata_dir = "./metadata"
    train_metadata_path = os.path.join(metadata_dir, "train_metadata.csv")
    val_metadata_path = os.path.join(metadata_dir, "val_metadata.csv")
    test_metadata_path = os.path.join(metadata_dir, "test_metadata.csv")

    # Working Directories (Write Access for idea_15)
    working_dir = os.path.join("./working", idea_name)
    output_dir = os.path.join(working_dir, "output")
    models_dir = os.path.join(working_dir, "models")
    cache_dir = os.path.join(working_dir, "cache")

    # Submission Directory
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Ensure all write-directories exist
    for d in [working_dir, output_dir, models_dir, cache_dir, submission_dir]:
        os.makedirs(d, exist_ok=True)

    # ==========================================
    # Data & Augmentation
    # ==========================================
    img_size = 256
    num_classes = 4
    target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    # Augmentation Strategy
    # Lesson 30: Vertical flips are crucial for rotationally invariant leaves
    vertical_flip = True
    horizontal_flip = True

    # Lesson 11: Mixup/CutMix caused underfitting in short regimes; explicitly disabled
    mixup = False
    cutmix = False

    # ==========================================
    # Model Architecture
    # ==========================================
    # Lesson 3/7/18: ResNet34 outperforms deeper/complex architectures on this dataset
    model_name = "resnet34"
    pretrained = True
    # Head: Standard GAP + FC (No GeM/Multi-Sample Dropout to avoid over-regularization)

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    batch_size = 32

    # Optimizer: Uniform Learning Rate (Lesson 44: Discriminative LRs caused instability)
    lr = 1e-4
    min_lr = 1e-6
    weight_decay = 1e-6

    # Scheduler: Cosine Annealing Warm Restarts
    # T_max will be synchronized with calibration_epochs

    # ==========================================
    # Execution Strategy
    # ==========================================

    # Phase 1: Proxy Calibration
    # Run 5-Fold CV to find the epoch where validation AUC peaks (E_opt)
    calibration_epochs = 20
    n_folds = 5

    # Phase 2: Production Training (Seed Ensemble)
    # Train 5 models on 100% of data for exactly E_opt epochs
    # Lesson 25: Full data training maximizes discriminative signal
    ensemble_seeds = [42, 2023, 777, 1990, 555]

    # Inference
    # Lesson 34: TTA degraded performance; disabled to ensure Validation-Inference Parity
    use_tta = False
