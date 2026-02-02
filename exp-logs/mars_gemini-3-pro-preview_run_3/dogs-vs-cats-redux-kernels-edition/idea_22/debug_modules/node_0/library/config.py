import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # Global Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    N_FOLDS = 5
    NUM_WORKERS = 12  # Optimizing for 12 vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_22"
    SUBMISSION_DIR = "./submission"

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    # Toggle DEBUG to True to run the pipeline on a small subset of data
    DEBUG = False
    DEBUG_SAMPLES = 200

    # -------------------------------------------------------------------------
    # Model Architecture & Training Configuration
    # -------------------------------------------------------------------------
    # Heterogeneous Ensemble Strategy:
    # 1. ResNet-50: Standard CNN, 256x256, 8 epochs
    # 2. ConvNeXt-Small: Modern CNN, 288x288, 8 epochs
    # 3. MaxViT-Tiny: Transformer, 224x224, 15 epochs

    MODELS = {
        "resnet50": {
            "model_name": "resnet50.a1_in1k",
            "img_size": 256,
            "batch_size": 64,
            "epochs": 8,
            "learning_rate": 1e-4,
            "min_lr": 1e-6,
            "weight_decay": 1e-2,
        },
        "convnext_small": {
            "model_name": "convnext_small.fb_in1k",
            "img_size": 288,
            "batch_size": 48,
            "epochs": 8,
            "learning_rate": 1e-4,
            "min_lr": 1e-6,
            "weight_decay": 1e-2,
        },
        "maxvit_tiny": {
            "model_name": "maxvit_tiny_tf_224.in1k",
            "img_size": 224,
            "batch_size": 32,
            "epochs": 15,
            "learning_rate": 5e-5,
            "min_lr": 1e-6,
            "weight_decay": 1e-2,
        },
    }

    # -------------------------------------------------------------------------
    # Inference & Calibration
    # -------------------------------------------------------------------------
    TTA_ENABLED = True  # Enable Test Time Augmentation (Horizontal Flip)
    # Threshold to discard models that fail to converge or perform poorly on OOF data
    # Set conservatively to avoid discarding viable models during initial runs
    OOF_LOSS_THRESHOLD = 0.5

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
