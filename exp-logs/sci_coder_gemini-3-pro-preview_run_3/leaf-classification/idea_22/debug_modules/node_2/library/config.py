import os


class Config:
    """
    Global configuration for the Leaf Classification Task.
    Strategy: Stratified Independent-Subspace Linear Discriminant Ensemble
              with Orthogonal Manifold Densification.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Specific working directory for this experiment (Idea 22)
    # This directory will store cached features, models, and temporary files.
    WORKING_DIR = "./working/idea_22"

    # Path for the final submission file
    SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # ==========================================
    # Data Processing & Augmentation
    # ==========================================
    # Number of equidistant rotations to extract features for.
    # 12 rotations = 30-degree increments (0, 30, 60, ..., 330).
    NUM_ROTATIONS = 12

    # Number of orthogonal centroids to generate per image.
    # This implements the Manifold Densification strategy.
    # Centroid A: Avg(0, 90, 180, 270)
    # Centroid B: Avg(30, 120, 210, 300)
    # Centroid C: Avg(60, 150, 240, 330)
    CENTROIDS_PER_IMAGE = 3

    # Target image size for neural network inputs
    IMG_SIZE = 224

    # ==========================================
    # Model Architectures (Feature Extractors)
    # ==========================================
    # Global Geometry Stream: DINOv2 (ViT-Large)
    # Captures self-supervised geometric priors.
    DINO_MODEL = "vit_large_patch14_dinov2"

    # Local Texture Stream: ConvNeXt Large
    # Captures high-frequency margin and texture details.
    CONVNEXT_MODEL = "convnext_large"

    # Batch size for feature extraction inference
    BATCH_SIZE = 32

    # Number of data loading workers (adjusted for 12 vCPUs)
    NUM_WORKERS = 4

    # ==========================================
    # Dimensionality Reduction & Classifier
    # ==========================================
    # Variance retention threshold for Independent PCA steps.
    # Applied separately to DINO and ConvNeXt features.
    PCA_VARIANCE = 0.99

    # ==========================================
    # Training & Cross-Validation
    # ==========================================
    # Number of folds for Stratified K-Fold Cross-Validation
    N_FOLDS = 10

    # Global random seed for reproducibility
    SEED = 42

    # Probability clipping values to avoid log-loss extremes
    PROB_CLIP_MIN = 1e-15
    PROB_CLIP_MAX = 1.0 - 1e-15

    # ==========================================
    # Debugging / Development
    # ==========================================
    # If set to an integer (e.g., 100), limits the dataset size for faster debugging.
    # If None, runs on the full dataset.
    DEBUG_SAMPLE_SIZE = None
