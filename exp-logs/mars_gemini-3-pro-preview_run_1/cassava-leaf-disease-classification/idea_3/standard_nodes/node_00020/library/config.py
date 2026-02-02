import os
import torch


class CFG:
    """
    Configuration class for Cassava Leaf Disease Classification.
    Defines hyperparameters, file paths, and model settings.
    """

    # ====================================================
    # General Settings
    # ====================================================
    seed = 42
    debug = False
    num_workers = 12  # Utilizing available vCPUs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_freq = 100

    # ====================================================
    # Data Paths
    # ====================================================
    # Input root directory
    input_root = "./input"

    # Metadata paths (pre-generated)
    metadata_dir = "./metadata"
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "val.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")

    # Output directories
    # Using idea_3 as the working directory for this experiment
    output_dir = "./working/idea_3"
    os.makedirs(output_dir, exist_ok=True)

    # Submission directory
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    # ====================================================
    # Model Architecture
    # ====================================================
    model_name = "convnext_small_in22k"
    num_classes = 5

    # Multi-Sample Dropout settings
    dropout_rate = 0.2
    num_dropout_samples = 8

    # ====================================================
    # Training Strategy (Progressive Resizing)
    # ====================================================
    # Image Resolutions
    img_size_base = 384  # Resolution for Warmup and Base training
    img_size_finetune = 512  # Resolution for Fine-tuning stage

    # Epochs per stage
    epochs_warmup = 1  # Frozen backbone, train head only
    epochs_base = 12  # Unfrozen backbone, base resolution
    epochs_finetune = 5  # Unfrozen backbone, high resolution

    # ====================================================
    # Optimization & Regularization
    # ====================================================
    # Batch Size
    batch_size = 32
    valid_batch_size = 64
    accum_iter = (
        1  # Gradient accumulation steps (Effective batch = batch_size * accum_iter)
    )

    # Optimizer (AdamW)
    learning_rate = 2e-4
    weight_decay = 1e-2
    min_lr = 1e-6

    # Scheduler (Cosine Annealing)
    # T_max will be set dynamically based on total steps in the training script

    # Augmentation (MixUp / CutMix)
    mixup_alpha = 0.8
    cutmix_alpha = 1.0
    mixup_prob = 0.5  # Probability of applying mixing regularization

    # Label Smoothing (if not using MixUp/CutMix for a batch)
    label_smoothing = 0.1

    # ====================================================
    # Inference
    # ====================================================
    tta_steps = 3  # Test Time Augmentation steps (e.g., Original + HFlip + VFlip)
