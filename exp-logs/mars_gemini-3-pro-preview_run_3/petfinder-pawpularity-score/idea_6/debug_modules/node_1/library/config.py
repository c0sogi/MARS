import os
import torch


class Config:
    """
    Configuration module for the Quad-Stream Semantic-Geometric Ensemble.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100

    # Compute
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use a moderate number of workers to balance overhead and throughput
    NUM_WORKERS = 4

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"

    # Metadata Files (Generated in previous steps)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_meta.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_meta.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_meta.csv")

    # Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working and output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Configuration
    # ==========================================
    IMAGE_SIZE = 224
    BATCH_SIZE = 32

    # Test-Time Augmentation: Average predictions of Original + Horizontal Flip
    USE_TTA = True

    # Metadata Features
    METADATA_COLS = [
        "Subject Focus",
        "Eyes",
        "Face",
        "Near",
        "Action",
        "Accessory",
        "Group",
        "Collage",
        "Human",
        "Occlusion",
        "Info",
        "Blur",
    ]

    # Metadata Amplification
    # Scale binary features by this factor to ensure they contribute meaningfully
    # to Euclidean distances in the SVR kernel (vs high-dim image embeddings).
    METADATA_SCALE = 10.0

    # ==========================================
    # Model Architecture (Quad-Stream)
    # ==========================================
    # We define the four backbones to be used for feature extraction.
    # Each dictionary specifies the model name, the library to load it from,
    # and the type of pre-training/signal it represents.
    BACKBONES = [
        # 1. Swin Transformer (Supervised)
        # Captures global layout and composition effectively.
        {
            "name": "swin_large_patch4_window7_224",
            "library": "timm",
            "type": "supervised",
        },
        # 2. EfficientNetV2 (Supervised)
        # Captures high-frequency details, texture, and image quality.
        {
            "name": "tf_efficientnetv2_l.in21k_ft_in1k",
            "library": "timm",
            "type": "supervised",
        },
        # 3. DINOv2 (Self-Supervised)
        # Captures object geometry and part-whole correspondence without label bias.
        {
            "name": "vit_large_patch14_dinov2.lvd142m",
            "library": "timm",
            "type": "self_supervised",
        },
        # 4. CLIP (Language-Aligned)
        # Captures semantic "vibe" and affective qualities (e.g., cuteness).
        {
            "name": "openai/clip-vit-large-patch14",
            "library": "transformers",
            "type": "clip",
        },
    ]

    # ==========================================
    # Feature Processing & Stacking
    # ==========================================
    # PCA Variance Retention: Compress each backbone's output independently
    # to retain this fraction of variance before concatenation.
    PCA_VARIANCE = 0.95

    # Cross-Validation Folds for Stacking
    N_FOLDS = 5
