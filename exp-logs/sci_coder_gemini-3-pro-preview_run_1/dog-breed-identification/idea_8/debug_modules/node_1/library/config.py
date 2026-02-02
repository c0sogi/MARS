import os
import torch


class Config:
    # --------------------
    # General Configuration
    # --------------------
    SEED = 42
    NUM_CLASSES = 120
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------
    # Directories
    # --------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORK_DIR = "./working/idea_8"

    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Output path for the final submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------
    # Data Pipeline
    # --------------------
    IMG_SIZE = 256  # Resize dimension
    CROP_SIZE = 224  # Center crop dimension
    BATCH_SIZE = 32  # Safe batch size for Base models on A100
    N_FOLDS = 5

    # --------------------
    # Model Architectures (timm)
    # --------------------
    # Expert A: ConvNeXt Base (Pre-trained on ImageNet-1k)
    # Good for local geometric features
    MODEL_A_NAME = "convnext_base.fb_in1k"

    # Expert B: Swin Transformer Base (Pre-trained on ImageNet-1k)
    # Good for global context
    MODEL_B_NAME = "swin_base_patch4_window7_224.ms_in1k"

    # --------------------
    # Training Regime A (ConvNeXt - Precision Track)
    # --------------------
    # Phase 1: Head Adaptation (Frozen Backbone)
    REGIME_A_PHASE1_EPOCHS = 1
    REGIME_A_PHASE1_LR = 1e-3

    # Phase 2: Fine-Tuning (Discriminative LRs)
    REGIME_A_PHASE2_EPOCHS = 6
    REGIME_A_BACKBONE_LR = 1e-6
    REGIME_A_HEAD_LR = 1e-4

    # Phase 3: Stochastic Weight Averaging (SWA)
    REGIME_A_SWA_EPOCHS = 3
    REGIME_A_SWA_LR = 1e-5

    # --------------------
    # Training Regime B (Swin - Robustness Track)
    # --------------------
    # Phase 1: Head Adaptation
    REGIME_B_PHASE1_EPOCHS = 1
    REGIME_B_PHASE1_LR = 1e-3

    # Phase 2: Regularized Fine-Tuning (Mixup/CutMix enabled)
    # Extended duration to allow convergence with soft targets
    REGIME_B_PHASE2_EPOCHS = 12
    REGIME_B_BASE_LR = 5e-5

    # Mixup/CutMix Hyperparameters
    MIXUP_ALPHA = 0.8
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 1.0

    # Phase 3: Cooldown (Augmentation Annealing - No Mixup)
    REGIME_B_COOLDOWN_EPOCHS = 3
    REGIME_B_COOLDOWN_LR = 1e-6

    # Phase 4: SWA
    REGIME_B_SWA_EPOCHS = 3
    REGIME_B_SWA_LR = 1e-6

    # --------------------
    # Stacking / Meta-Learner
    # --------------------
    META_SOLVER_C = 1.0

    @classmethod
    def setup(cls):
        """Ensures necessary working directories exist."""
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
