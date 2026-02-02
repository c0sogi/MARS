import os
import torch


class Config:
    # =========================================================================
    # General Configuration
    # =========================================================================
    seed = 42
    num_workers = 12
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Data Paths
    # =========================================================================
    # Input directories (Read-Only)
    input_dir = "./input"
    train_images_dir = os.path.join(input_dir, "train_images")
    test_images_dir = os.path.join(input_dir, "test_images")

    # Metadata directories (Generated previously)
    metadata_dir = "./metadata"
    train_metadata_path = os.path.join(metadata_dir, "train.csv")
    val_metadata_path = os.path.join(metadata_dir, "val.csv")
    test_metadata_path = os.path.join(metadata_dir, "test.csv")

    # Output directories (Writeable)
    # Using 'idea_8' as the specific working directory for this iteration
    working_dir = "./working/idea_8"
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_name = "convnext_small.fb_in1k"
    num_classes = 5
    drop_rate = 0.0
    drop_path_rate = 0.4  # Stochastic Depth

    # Model EMA (Exponential Moving Average)
    use_ema = True
    model_ema_decay = 0.9999

    # =========================================================================
    # Training Configuration (Progressive Resolution)
    # =========================================================================
    n_folds = 5

    # Optimizer
    learning_rate = 2e-4
    weight_decay = 0.05
    min_lr = 1e-6

    # Loss
    label_smoothing = 0.1

    # --- Phase 1: Coarse Learning (Global Features) ---
    # Resolution: 224x224
    # Augmentation: Stronger Mixing
    phase1_epochs = 8
    phase1_image_size = 224
    phase1_batch_size = 64
    phase1_mixup_prob = 0.5

    # --- Phase 2: Fine-Tuning (Fine-Grained Details) ---
    # Resolution: 384x384
    # Augmentation: Weaker Mixing to resolve details
    phase2_epochs = 4
    phase2_image_size = 384
    phase2_batch_size = 32  # Reduced batch size for larger images
    phase2_accum_iter = 1  # Gradient accumulation steps if needed
    phase2_mixup_prob = 0.2

    # MixUp / CutMix Hyperparameters
    mixup_alpha = 0.8
    cutmix_alpha = 1.0

    # =========================================================================
    # Inference Configuration
    # =========================================================================
    # Test Time Augmentation (TTA)
    tta_flips = True  # Horizontal Flip

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        """
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)

    @classmethod
    def get_phase_config(cls, phase):
        """
        Returns a dictionary of configuration specific to the training phase.
        phase: int (1 or 2)
        """
        if phase == 1:
            return {
                "epochs": cls.phase1_epochs,
                "image_size": cls.phase1_image_size,
                "batch_size": cls.phase1_batch_size,
                "mixup_prob": cls.phase1_mixup_prob,
                "accum_iter": 1,
            }
        elif phase == 2:
            return {
                "epochs": cls.phase2_epochs,
                "image_size": cls.phase2_image_size,
                "batch_size": cls.phase2_batch_size,
                "mixup_prob": cls.phase2_mixup_prob,
                "accum_iter": cls.phase2_accum_iter,
            }
        else:
            raise ValueError("Phase must be 1 or 2")


# Initialize directories on import
Config.setup()
