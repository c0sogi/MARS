import os
import torch


class CFG:
    """
    Configuration class for Cassava Leaf Disease Classification pipeline.
    Centralizes all hyperparameters, paths, and model settings.
    """

    # ================= General Settings =================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    num_workers = 4  # Adjust based on available vCPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_freq = 100  # Logging frequency in steps

    # ================= Directories & Paths =================
    # Input Data (Read-Only)
    input_root = "./input"
    metadata_dir = "./metadata"

    # Metadata Files
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "val.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")

    # Output Directories
    working_dir = "./working/idea_6"
    output_dir = os.path.join(working_dir, "outputs")
    checkpoint_dir = os.path.join(working_dir, "checkpoints")

    # Submission
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # ================= Model Architecture =================
    # Backbone
    model_name = "convnext_small_384_in22ft1k"
    num_classes = 5

    # Structural Innovations
    use_gem = True  # Generalized Mean Pooling
    gem_p = 3.0  # Initial p value for GeM
    gem_learnable = True  # Whether p is a learnable parameter

    use_msd = True  # Multi-Sample Dropout Head
    msd_num = 5  # Number of dropout samples
    msd_rate = 0.5  # Dropout rate

    # Regularization
    drop_path_rate = 0.0  # Stochastic Depth rate (Disabled to maximize capacity)

    # ================= Data Pipeline =================
    # Progressive Resizing
    img_size_p1 = 384  # Image size for Phase 1 (Base Training)
    img_size_p2 = 512  # Image size for Phase 2 (Fine-tuning & SWA)

    # Augmentation
    mixup_alpha = 0.8
    cutmix_alpha = 1.0
    mixup_prob = 1.0  # Probability of applying MixUp or CutMix
    crop_scale = (0.3, 1.0)  # Scale range for RandomResizedCrop

    # ================= Training Strategy =================
    # Batch Size & Optimization
    batch_size = 8  # Physical batch size per GPU
    grad_accumulation = 4  # Gradient accumulation steps
    # Effective Batch Size = 8 * 4 = 32

    # Learning Rate & Scheduler
    lr = 2e-4  # Base learning rate
    min_lr = 1e-6  # Minimum learning rate for scheduler
    weight_decay = 1e-2
    llrd_decay = 0.8  # Layer-wise Learning Rate Decay factor

    # Training Schedule (Epochs)
    epochs_warmup = 1  # Epochs for warming up head/GeM with frozen backbone
    epochs_base = 8  # Epochs for base training at 384x384
    epochs_finetune = 4  # Epochs for high-res fine-tuning at 512x512
    epochs_swa = 4  # Epochs for Stochastic Weight Averaging at 512x512

    # Loss
    label_smoothing = 0.0  # Explicit label smoothing (0.0 because MixUp is used)

    # ================= Inference =================
    tta = True  # Enable Test Time Augmentation (Horizontal/Vertical Flip)

    @classmethod
    def setup(cls):
        """
        Ensures all necessary output directories exist.
        """
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)
        os.makedirs(cls.checkpoint_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)
