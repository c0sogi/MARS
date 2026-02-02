import os


class Config:
    """
    Configuration module for the Leaf Species Identification pipeline.
    Implements the strategy: Selective-Topology Orthogonal Manifold-Densified LDA.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Working directory for caching intermediate results
    # Using 'idea_32' as the designated workspace
    WORKING_DIR = "./working/idea_32"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Final Submission Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Global & Debug Parameters
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4  # Workers for data loading

    # Debugging / Quick Run parameters
    # Set DEBUG to True to limit dataset size for rapid testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50

    # ==========================================
    # Feature Extraction Parameters
    # ==========================================
    IMAGE_SIZE = 224
    BATCH_SIZE = 32

    # Models
    # DINOv2 Large (ViT-Large) - Captures Global Geometric Priors
    MODEL_DINOV2 = "vit_large_patch14_dinov2.lvd142m"
    # ConvNeXt Large - Captures Fine-scale Margin/Texture details
    MODEL_CONVNEXT = "convnext_large.fb_in22k_ft_in1k"

    # ==========================================
    # Manifold Densification (Rotations)
    # ==========================================
    # We extract 12 equidistant views: 0, 30, 60, ..., 330 degrees
    ROTATION_ANGLES = list(range(0, 360, 30))

    # Orthogonal Centroid Definitions
    # We group the 12 views into 3 orthogonal centroids (A, B, C).
    # Each centroid is an average of 4 mutually exclusive orthogonal views.
    # Indices correspond to the ROTATION_ANGLES list.
    # Centroid A: {0, 90, 180, 270}   -> Indices [0, 3, 6, 9]
    # Centroid B: {30, 120, 210, 300} -> Indices [1, 4, 7, 10]
    # Centroid C: {60, 150, 240, 330} -> Indices [2, 5, 8, 11]
    CENTROID_INDICES = [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]

    # ==========================================
    # Pipeline / Training Parameters
    # ==========================================
    # PCA Variance retention for Independent Subspace Reduction (Visual Streams)
    PCA_VARIANCE = 0.99

    # Cross-Validation Strategy (Stratified K-Fold)
    N_FOLDS = 10

    # Probability Clipping to avoid log loss extremes
    # Predicted probabilities are replaced with max(min(p, 1-eps), eps)
    PROB_CLIP_EPS = 1e-15

    # ==========================================
    # Cache Filenames
    # ==========================================
    # These files will be stored in CACHE_DIR

    # Extracted Image Features (Shape: [N, 12, Feature_Dim])
    CACHE_TRAIN_IMG_FEATURES = "train_img_features.npy"
    CACHE_TEST_IMG_FEATURES = "test_img_features.npy"

    # Tabular Features (Shape: [N, 192])
    CACHE_TRAIN_TAB_FEATURES = "train_tab_features.npy"
    CACHE_TEST_TAB_FEATURES = "test_tab_features.npy"

    # Identifiers and Labels
    CACHE_TRAIN_IDS = "train_ids.npy"
    CACHE_TEST_IDS = "test_ids.npy"
    CACHE_TRAIN_LABELS = "train_labels.npy"
