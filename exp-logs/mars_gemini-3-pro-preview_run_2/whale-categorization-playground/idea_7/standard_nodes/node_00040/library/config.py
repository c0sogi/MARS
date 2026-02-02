import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use available vCPUs for data loading
    NUM_WORKERS = os.cpu_count() if os.cpu_count() is not None else 4

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Sample submission
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output / Working Directory
    # We use 'idea_7' as the designated workspace for this solution
    WORKING_DIR = "./working/idea_7"
    os.makedirs(WORKING_DIR, exist_ok=True)

    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Model Architecture & Hyperparameters
    # -------------------------------------------------------------------------
    # Dual-Backbone Ensemble Strategy
    # We train two structurally different models to maximize ensemble diversity.
    MODEL_CONFIGS = [
        {"backbone": "tf_efficientnet_b4", "name": "model_a", "embedding_size": 512},
        {"backbone": "tf_efficientnetv2_m", "name": "model_b", "embedding_size": 512},
    ]

    # ArcFace Head Parameters
    # s=30.0 is chosen to avoid convergence issues seen with higher scales on small datasets
    ARCFACE_SCALE = 30.0
    ARCFACE_MARGIN = 0.5

    # Number of classes (Known whales only)
    # Total unique IDs in train is 4029. We exclude 'new_whale' during training.
    # 4029 - 1 = 4028.
    N_CLASSES = 4028

    # -------------------------------------------------------------------------
    # Training Loop Parameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32  # Strict lower limit to maintain BatchNorm stability
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Progressive Resizing Stages
    # Stage 1: Fast learning of structural features at 256x256
    # Stage 2: Fine-tuning on fine-grained details at 384x384
    STAGES = [{"resolution": 256, "epochs": 15}, {"resolution": 384, "epochs": 15}]

    # Gradient Checkpointing
    # Essential for fitting B4/V2-M at 384px on GPU memory
    USE_GRADIENT_CHECKPOINTING = True

    # -------------------------------------------------------------------------
    # Inference Parameters
    # -------------------------------------------------------------------------
    # Distance Thresholding for Open-Set Recognition
    # If cosine similarity < threshold, predict 'new_whale'
    CONFIDENCE_THRESHOLD = 0.5

    # Test-Time Augmentation
    # Enables Horizontal Flip TTA during inference
    USE_TTA = True

    # -------------------------------------------------------------------------
    # Caching Utilities
    # -------------------------------------------------------------------------
    @staticmethod
    def get_cache_path(identifier: str) -> str:
        """
        Generates a safe path for caching numpy arrays in the working directory.

        Args:
            identifier: Unique string for the data (e.g., 'train_images_256')

        Returns:
            Full path to the .npy file.
        """
        filename = f"{identifier}.npy"
        return os.path.join(Config.WORKING_DIR, filename)


# Apply seeding immediately
seed_everything(Config.SEED)
