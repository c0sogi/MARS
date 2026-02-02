import os
import torch


class Config:
    """
    Configuration for the Dog vs Cat Classification Task.
    Implements a Stratified K-Fold Heterogeneous Ensemble with Progressive Resizing.
    """

    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Using 4 workers is generally safe and efficient for the provided vCPU count
    NUM_WORKERS = 4

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_21"
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    N_FOLDS = 5
    OPTIMIZER_NAME = "AdamW"
    WEIGHT_DECAY = 1e-2
    SCHEDULER_NAME = "CosineAnnealingLR"
    LOSS_FN = "BCEWithLogitsLoss"
    MAX_GRAD_NORM = 1.0

    # Augmentation Settings (Context-Preserving)
    AUG_RRC_SCALE = (0.8, 1.0)  # RandomResizedCrop scale to prevent subject loss
    AUG_COLOR_JITTER = 0.2  # Intensity for brightness, contrast, saturation

    # -------------------------------------------------------------------------
    # Model Architectures & Progressive Resizing Schedules
    # -------------------------------------------------------------------------
    # The 'Golden Trio' for maximum inductive bias diversity
    MODELS = {
        "resnet50": {
            "backbone": "resnet50.a1_in1k",
            "batch_size": 64,
            "lr": 1e-4,
            "min_lr": 1e-6,
            "phases": [
                # Phase 1: Speed (Low Res)
                {"img_size": 192, "epochs": 4},
                # Phase 2: Accuracy (Target Res)
                {"img_size": 256, "epochs": 4},
            ],
        },
        "convnext_small": {
            "backbone": "convnext_small.fb_in1k",
            "batch_size": 48,
            "lr": 1e-4,
            "min_lr": 1e-6,
            "phases": [
                # Phase 1: Speed
                {"img_size": 224, "epochs": 4},
                # Phase 2: Accuracy (Target Res: 288 for ConvNeXt)
                {"img_size": 288, "epochs": 4},
            ],
        },
        "maxvit_tiny": {
            "backbone": "maxvit_tiny_tf_224.in1k",
            "batch_size": 32,
            "lr": 5e-5,  # Transformers typically require lower LR
            "min_lr": 1e-6,
            "phases": [
                # Phase 1: Speed
                {"img_size": 224, "epochs": 7},
                # Phase 2: Accuracy
                {"img_size": 224, "epochs": 8},
            ],
        },
    }

    # -------------------------------------------------------------------------
    # Inference & Calibration
    # -------------------------------------------------------------------------
    USE_TTA = True  # Use Horizontal Flip Test Time Augmentation
    OOF_THRESHOLD = 0.05  # Discard models with OOF Log Loss > 0.05
    CALIBRATION_METHOD = "IsotonicRegression"

    @classmethod
    def setup(cls, debug=False):
        """
        Initializes necessary directories and adjusts configuration for debugging.

        Args:
            debug (bool): If True, reduces folds and epochs for rapid testing.
        """
        # Ensure output directories exist
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        if debug:
            print(f"DEBUG MODE ACTIVATED: Reducing folds and epochs.")
            cls.N_FOLDS = 2
            for model_name in cls.MODELS:
                # Reduce to 1 epoch per phase for fast pipeline verification
                for phase in cls.MODELS[model_name]["phases"]:
                    phase["epochs"] = 1
