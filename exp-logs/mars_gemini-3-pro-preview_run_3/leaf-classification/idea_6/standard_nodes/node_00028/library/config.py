import os
import torch


class Config:
    """
    Global configuration for the Plant Species Classification pipeline.
    Implements the 'Homogeneous Bagged Ensemble of LDA with Multi-View Self-Supervised Features' strategy.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Directories & Paths
    # ==========================================
    # Read-only input directories
    INPUT_DIR = "./input"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Sample Submission
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Writeable working directories
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_6")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing & Feature Extraction
    # ==========================================
    # Image parameters
    IMAGE_SIZE = 224  # Standard input size for DINOv2 and ConvNeXt
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Dataloader parameters
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Model Architectures (timm library names)
    # Global Geometry Stream: DINOv2 (ViT-Large)
    MODEL_DINOV2 = "vit_large_patch14_dinov2.lvd142m"
    # Local Texture Stream: ConvNeXt Large
    MODEL_CONVNEXT = "convnext_large.fb_in22k_ft_in1k"

    # ==========================================
    # Dimensionality Reduction & Classifier
    # ==========================================
    # PCA Variance Retention
    PCA_VARIANCE = 0.99

    # Ensemble Strategy (Bagging LDA)
    N_ESTIMATORS = 20

    # Probability Clipping (as per metric requirements)
    PROB_EPSILON = 1e-15

    # ==========================================
    # Debugging & Development
    # ==========================================
    # Set to a small integer (e.g., 100) to limit dataset size during debugging
    # Set to None for full run
    MAX_SAMPLES = None

    @classmethod
    def setup(cls):
        """
        Ensures necessary writeable directories exist.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def print_config(cls):
        """
        Prints the current configuration setup.
        """
        print("=" * 40)
        print("CONFIG SETUP")
        print("=" * 40)
        print(f"Device:         {cls.DEVICE}")
        print(f"Seed:           {cls.SEED}")
        print(f"Image Size:     {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"Batch Size:     {cls.BATCH_SIZE}")
        print(f"Backbones:      {cls.MODEL_DINOV2}, {cls.MODEL_CONVNEXT}")
        print(f"PCA Variance:   {cls.PCA_VARIANCE}")
        print(f"Ensemble Size:  {cls.N_ESTIMATORS}")
        print(f"Cache Dir:      {cls.CACHE_DIR}")
        print(f"Output Path:    {cls.SUBMISSION_PATH}")
        print("=" * 40)
