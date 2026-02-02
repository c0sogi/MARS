import os
import torch


class Config:
    """
    Configuration module for the Leaf Classification task.
    Stores global constants, hyperparameters, and path definitions for the
    Cross-Validated Manifold-Densified LDA Ensemble solution.
    """

    # ==========================================
    # Global Execution Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Optimized for the available 12 vCPUs
    NUM_WORKERS = 4

    # Debugging Control
    # Set to True to run on a small subset of data for testing pipeline logic
    DEBUG = False
    DEBUG_SAMPLES = 50

    # ==========================================
    # Directory & File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_14"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist immediately
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Data Paths
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    CACHE_DIR = WORKING_DIR

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    # Image Preprocessing
    IMAGE_SIZE = 224
    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)

    # Manifold Densification (Orthogonal View-Set Augmentation)
    # We generate 12 views by rotating the image every 30 degrees.
    ROTATION_STEP = 30
    ROTATION_ANGLES = list(range(0, 360, ROTATION_STEP))

    # View Indices for Centroid Generation
    # We form 3 centroids per image in training by averaging orthogonal views.
    # Set A: {0, 90, 180, 270} -> Indices [0, 3, 6, 9]
    # Set B: {30, 120, 210, 300} -> Indices [1, 4, 7, 10]
    # Set C: {60, 150, 240, 330} -> Indices [2, 5, 8, 11]
    VIEW_INDICES_A = [0, 3, 6, 9]
    VIEW_INDICES_B = [1, 4, 7, 10]
    VIEW_INDICES_C = [2, 5, 8, 11]

    # ==========================================
    # Model Architecture & Training
    # ==========================================
    # Feature Extractors (timm model names)
    # Global Geometry Stream: DINOv2 (ViT-Large)
    MODEL_DINO_NAME = "vit_large_patch14_dinov2.lvd142m"
    # Local Texture Stream: ConvNeXt Large
    MODEL_CONV_NAME = "convnext_large.fb_in22k_ft_in1k"

    # Batch size for feature extraction (inference mode)
    BATCH_SIZE = 32

    # Independent Subspace Reduction
    # Retain 99% variance for each stream independently
    PCA_VARIANCE = 0.99

    # Tabular Feature Transformation
    # Enforce Gaussian distribution for LDA
    TABULAR_OUTPUT_DIST = "normal"

    # Ensemble Strategy
    N_FOLDS = 10

    # LDA Classifier Settings
    # 'lsqr' with 'auto' shrinkage approximates Ledoit-Wolf estimation
    LDA_SOLVER = "lsqr"
    LDA_SHRINKAGE = "auto"
