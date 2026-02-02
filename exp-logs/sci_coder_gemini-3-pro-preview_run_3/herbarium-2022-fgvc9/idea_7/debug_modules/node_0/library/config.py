import os
import torch


class Config:
    """
    Configuration for Hierarchical EfficientNetV2 Plant Classification.
    Encapsulates hyperparameters for Progressive Resolution training and Multi-Task Learning.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 5000

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files (pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Raw JSON for hierarchy extraction
    TRAIN_METADATA_JSON = os.path.join(INPUT_DIR, "train_metadata.json")

    # Working directory for this specific experiment (Idea 7)
    WORK_DIR = "./working/idea_7"
    os.makedirs(WORK_DIR, exist_ok=True)

    # Cache file for taxonomic hierarchy mappings (Species -> Genus -> Family)
    # Using parquet as requested for caching
    HIERARCHY_CACHE_PATH = os.path.join(WORK_DIR, "hierarchy_mappings.parquet")

    # Checkpoint directories for the two training stages
    CHECKPOINT_DIR_STAGE1 = os.path.join(WORK_DIR, "stage_1")
    CHECKPOINT_DIR_STAGE2 = os.path.join(WORK_DIR, "stage_2")
    os.makedirs(CHECKPOINT_DIR_STAGE1, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR_STAGE2, exist_ok=True)

    # Submission output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data & Hierarchy Specifications
    # =========================================================================
    NUM_CLASSES = 15501  # Species (Primary Task)

    # Auxiliary Tasks (Approximate counts, to be verified via hierarchy mapping)
    NUM_GENERA = 2564
    NUM_FAMILIES = 272

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # EfficientNetV2-Small: High parameter efficiency, fast convergence
    # Using variant pretrained on ImageNet-21k and finetuned on 1k
    MODEL_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"

    # Generalized Mean Pooling to focus on salient plant features
    USE_GEM_POOLING = True

    # Regularization
    DROPOUT_RATE = 0.2
    DROP_PATH_RATE = 0.2

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # Optimization
    OPTIMIZER = "AdamW"
    WEIGHT_DECAY = 0.01
    LR_MAX = 1e-3  # Max LR for OneCycle Scheduler
    LABEL_SMOOTHING = 0.1

    # Multi-Task Loss Weights
    # L_total = L_species + 0.1 * L_genus + 0.1 * L_family
    WEIGHT_SPECIES = 1.0
    WEIGHT_GENUS = 0.1
    WEIGHT_FAMILY = 0.1

    # --- Stage 1: Feature Learning ---
    # Lower resolution, larger batch size for rapid convergence
    STAGE1_IMG_SIZE = 224
    STAGE1_BATCH_SIZE = 128
    STAGE1_EPOCHS = 12

    # --- Stage 2: Fine-Grained Refinement ---
    # Higher resolution to resolve fine details (e.g., venation, texture)
    STAGE2_IMG_SIZE = 384
    STAGE2_BATCH_SIZE = 32  # Reduced to accommodate higher resolution on GPU
    STAGE2_EPOCHS = 8

    # =========================================================================
    # Augmentation Strategy
    # =========================================================================
    # Strong augmentation (RandomResizedCrop, ColorJitter)
    # Mixup/CutMix are explicitly excluded
    AUG_SCALE = (0.08, 1.0)
    AUG_RATIO = (3.0 / 4.0, 4.0 / 3.0)
    COLOR_JITTER = 0.4

    # =========================================================================
    # Inference
    # =========================================================================
    # Test Time Augmentation
    TTA_FLIP = True  # Average predictions of image and its horizontal flip
