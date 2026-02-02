import os


class Config:
    """
    Configuration module for Manifold-Densified Linear Discriminant Analysis pipeline.
    Defines paths, hyperparameters, and constants for the leaf classification task.
    """

    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching intermediate features (idea_13)
    WORKING_DIR = "./working/idea_13"

    # Output submission directory and file
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Data Processing Parameters
    # ==========================================
    # Standard image size for model input
    IMG_SIZE = 224

    # Multi-View Augmentation: 12 equidistant views
    # Angles: 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330
    ROTATION_ANGLES = list(range(0, 360, 30))

    # Manifold Densification Groups (Orthogonal View-Sets)
    # We generate 3 training samples per image by averaging these view groups.
    # Group A is also used for Inference.
    VIEW_GROUPS = [
        [0, 90, 180, 270],  # Group A (Indices: 0, 3, 6, 9)
        [30, 120, 210, 300],  # Group B (Indices: 1, 4, 7, 10)
        [60, 150, 240, 330],  # Group C (Indices: 2, 5, 8, 11)
    ]

    # Inference uses only the standard orthogonal set (Group A)
    INFERENCE_ANGLES = [0, 90, 180, 270]

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Feature Extractor Backbones (timm library tags)
    # Global Geometry Stream: DINOv2 (ViT-Large)
    MODEL_DINOV2 = "vit_large_patch14_dinov2.lvd142m"

    # Local Texture Stream: ConvNeXt Large
    MODEL_CONVNEXT = "convnext_large.fb_in22k_ft_in1k"

    # Extraction parameters
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Dimensionality Reduction
    PCA_VARIANCE = 0.99

    # Classifier: Linear Discriminant Analysis
    LDA_SOLVER = "lsqr"
    LDA_SHRINKAGE = "auto"

    # ==========================================
    # Metric & Submission
    # ==========================================
    # Probability clipping to avoid log loss extremes
    PROB_CLIP_MIN = 1e-15
    PROB_CLIP_MAX = 1.0 - 1e-15

    @staticmethod
    def setup():
        """
        Initializes the working environment by ensuring necessary directories exist.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
