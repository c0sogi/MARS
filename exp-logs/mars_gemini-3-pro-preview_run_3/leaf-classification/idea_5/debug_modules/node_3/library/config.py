import os
import torch


class Config:
    """
    Global configuration for the Bagged Dual-Stream Generative Classifier.
    """

    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Specific working directory for this idea to store cached features/models
    WORKING_DIR = "./working/idea_5"

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Final Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Model Architectures & Image Settings
    # ==========================================
    # Stream 1: Self-Supervised Shape (DINOv2 ViT-Large)
    # Using a specific tag to ensure reproducibility
    MODEL_SHAPE_NAME = "vit_large_patch14_dinov2.lvd142m"
    IMG_SIZE_SHAPE = 518

    # Stream 2: Supervised Texture (ConvNeXt Large)
    # Using specific tag for ImageNet-22k pretraining fine-tuned on 1k
    MODEL_TEXTURE_NAME = "convnext_large.fb_in22k_ft_in1k"
    IMG_SIZE_TEXTURE = 384

    # Normalization constants (Standard ImageNet)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ==========================================
    # 3. Pipeline Hyperparameters
    # ==========================================
    SEED = 42

    # Dimensionality Reduction
    PCA_VARIANCE = 0.99

    # Classification / Bagging
    N_FOLDS = 10

    # Tabular Data
    # 192 features: 64 margin + 64 shape + 64 texture
    TABULAR_COLS_PREFIXES = ["margin", "shape", "texture"]

    # ==========================================
    # 4. Compute & Execution
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Batch size for feature extraction
    # A100 40GB can handle reasonable batch sizes for these large models
    BATCH_SIZE = 32
    NUM_WORKERS = 2

    # Debug flag to run on a subset of data
    DEBUG = False

    # ==========================================
    # 5. Caching Filenames
    # ==========================================
    # Standardized names for cached numpy arrays in WORKING_DIR

    # Training Data
    CACHE_TRAIN_FEATS_SHAPE = os.path.join(WORKING_DIR, "train_feats_shape.npy")
    CACHE_TRAIN_FEATS_TEXTURE = os.path.join(WORKING_DIR, "train_feats_texture.npy")
    CACHE_TRAIN_TABULAR = os.path.join(WORKING_DIR, "train_tabular.npy")
    CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")

    # Validation Data
    CACHE_VAL_FEATS_SHAPE = os.path.join(WORKING_DIR, "val_feats_shape.npy")
    CACHE_VAL_FEATS_TEXTURE = os.path.join(WORKING_DIR, "val_feats_texture.npy")
    CACHE_VAL_TABULAR = os.path.join(WORKING_DIR, "val_tabular.npy")
    CACHE_VAL_LABELS = os.path.join(WORKING_DIR, "val_labels.npy")

    # Test Data
    CACHE_TEST_FEATS_SHAPE = os.path.join(WORKING_DIR, "test_feats_shape.npy")
    CACHE_TEST_FEATS_TEXTURE = os.path.join(WORKING_DIR, "test_feats_texture.npy")
    CACHE_TEST_TABULAR = os.path.join(WORKING_DIR, "test_tabular.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # Model/Pipeline Artifacts
    # We will save the fitted pipeline (PCA + LDA) objects using joblib/pickle
    # Since we have N_FOLDS, we will name them dynamically in the training script
    # e.g., os.path.join(WORKING_DIR, f"pipeline_fold_{i}.pkl")
    PIPELINE_FILENAME_TEMPLATE = os.path.join(WORKING_DIR, "pipeline_fold_{}.pkl")
