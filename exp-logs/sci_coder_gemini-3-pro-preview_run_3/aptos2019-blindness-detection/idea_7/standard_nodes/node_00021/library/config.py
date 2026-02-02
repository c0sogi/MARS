import os
import torch


class Config:
    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    N_FOLDS = 5
    NUM_CLASSES = 1  # Regression output
    BACKBONE = "convnext_base"  # ConvNeXt Base backbone
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Compute resources
    # We have 12 vCPUs, so we can use a good number of workers
    NUM_WORKERS = 8

    # ==========================================
    # File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files (Pre-stratified)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for this specific idea (Idea 7)
    WORKING_DIR = "./working/idea_7"

    # Cache directory for processed numpy arrays (npy)
    # This is where we store the 512x512 and 1024x1024 processed images
    CACHE_DIR = WORKING_DIR

    # Directory to save trained model weights
    MODEL_DIR = os.path.join(WORKING_DIR, "models")

    # Final submission path
    SUBMISSION_PATH = "./submission/submission.csv"

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Set DEBUG to True to run on a small subset of data for quick verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode

    # ==========================================
    # Progressive Training Strategy
    # ==========================================

    # --- Stage 1: Structure Learning (Low Res) ---
    STAGE1_IMG_SIZE = 512
    STAGE1_BATCH_SIZE = 32  # Fits comfortably on A100 40GB
    STAGE1_EPOCHS = 10  # Sufficient for convergence on structure
    STAGE1_LR = 1e-4  # Standard starting LR

    # --- Stage 2: Fine-Grained Adaptation (High Res) ---
    STAGE2_IMG_SIZE = 1024
    STAGE2_BATCH_SIZE = 2  # Small physical batch size for high res
    STAGE2_GRAD_ACCUM = 16  # 2 * 16 = 32 effective batch size
    STAGE2_EPOCHS = 5  # Fine-tuning epochs
    STAGE2_LR = 1e-5  # Lower LR for fine-tuning

    # ==========================================
    # Inference
    # ==========================================
    INFERENCE_IMG_SIZE = 1024  # Match Stage 2 resolution
    INFERENCE_BATCH_SIZE = 4  # Slightly larger than training batch since no gradients

    @classmethod
    def setup(cls):
        """
        Creates the necessary working directories.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        # Submission directory is usually handled by the environment,
        # but we ensure the parent dir exists just in case
        os.makedirs(os.path.dirname(cls.SUBMISSION_PATH), exist_ok=True)
