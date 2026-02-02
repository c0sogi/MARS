import os
import torch


class CFG:
    """
    Configuration for Cassava Leaf Disease Classification - Idea 14

    Strategy:
    - Backbone: ConvNeXt Base (High Capacity)
    - Training: 5-Fold Stratified Cross-Validation
    - Curriculum:
        - Phase 1: 224x224, 12 Epochs (Coarse features)
        - Phase 2: 384x384, 8 Epochs (Fine-grained features)
    - Regularization: MixUp/CutMix (p=0.5), Stochastic Depth (0.5), Model EMA
    - Hardware: A100 40GB (Target Effective Batch Size: 32)
    """

    # =======================
    # General / Infrastructure
    # =======================
    seed = 42
    num_workers = 12
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging flags to control dataset size and runtime for testing
    debug = False
    debug_sample_size = 500  # Number of samples to use if debug=True

    # =======================
    # Directories & Paths
    # =======================
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Output directory specific to this experiment idea
    working_dir = "./working/idea_14"
    output_dir = working_dir

    # Metadata paths (Pre-generated)
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "val.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")

    # Submission output
    submission_csv = os.path.join(working_dir, "submission.csv")

    # =======================
    # Model Configuration
    # =======================
    model_name = "convnext_base.fb_in1k"
    num_classes = 5

    # Regularization for High-Capacity Model
    drop_path_rate = 0.5

    # Exponential Moving Average
    use_ema = True
    ema_decay = 0.9999

    # =======================
    # Training Configuration
    # =======================
    n_folds = 5

    # Optimization
    lr = 1e-4
    min_lr = 1e-6
    weight_decay = 0.05

    # Effective Batch Size (Gradient Accumulation Target)
    target_batch_size = 32

    # --- Phase 1: Coarse Feature Learning ---
    p1_img_size = 224
    p1_epochs = 12
    # Physical batch size for Phase 1 (fits easily in A100)
    p1_batch_size = 32

    # --- Phase 2: High-Fidelity Fine-Tuning ---
    p2_img_size = 384
    p2_epochs = 8
    # Physical batch size for Phase 2 (reduced to fit memory)
    # Gradient accumulation will be used to match target_batch_size (32)
    p2_batch_size = 16

    # =======================
    # Augmentation
    # =======================
    # MixUp and CutMix settings
    # Note: Idea 14 explicitly requires maintaining these during Phase 2
    mixup_p = 0.5
    cutmix_p = 0.5
    mixup_alpha = 0.8
    cutmix_alpha = 1.0

    # =======================
    # Inference
    # =======================
    # Test Time Augmentation
    tta = True  # Enables Horizontal Flip TTA

    @classmethod
    def setup(cls):
        """Ensures the working directory exists."""
        os.makedirs(cls.working_dir, exist_ok=True)


# Execute setup on module import
CFG.setup()
