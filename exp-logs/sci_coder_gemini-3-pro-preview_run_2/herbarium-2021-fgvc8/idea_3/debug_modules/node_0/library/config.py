import os
import torch


class Config:
    """
    Configuration for the Hierarchical ConvNeXt-Tiny Plant Classification Pipeline.
    This configuration supports a two-stage training process:
    1. Representation Learning (Instance-Balanced)
    2. Classifier Re-balancing (Class-Balanced)
    """

    # ==========================================
    # Directories & File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific experiment (Idea 3)
    # Used for checkpoints and intermediate cache files
    WORK_DIR = "./working/idea_3"
    os.makedirs(WORK_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Metadata Files (Raw JSONs)
    TRAIN_METADATA_JSON = os.path.join(INPUT_DIR, "train/metadata.json")
    TEST_METADATA_JSON = os.path.join(INPUT_DIR, "test/metadata.json")

    # Pre-processed CSV Splits (generated in previous steps)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Cache for Taxonomy Mappings (Category ID -> Family/Order)
    # This parquet file will store the hierarchical relationships derived from metadata
    TAXONOMY_MAP_PATH = os.path.join(WORK_DIR, "taxonomy_mappings.parquet")

    # Model Checkpoints
    STAGE1_CHECKPOINT = os.path.join(WORK_DIR, "stage1_checkpoint.pth")
    BEST_MODEL_PATH = os.path.join(WORK_DIR, "best_model.pth")

    # Final Submission Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    NUM_WORKERS = 12
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debug flag: Set to an integer (e.g., 5000) to limit dataset size for testing pipeline
    # Set to None for full production run
    DEBUG_SAMPLE_SIZE = None

    # ==========================================
    # Data & Model Hyperparameters
    # ==========================================
    # Image Resolution: 224x224 is standard for ConvNeXt and efficient for high throughput
    IMG_SIZE = 224

    # Batch Size: 128 fits comfortably on A100 40GB with ConvNeXt-Tiny
    BATCH_SIZE = 128

    # Model Architecture
    BACKBONE = "convnext_tiny"
    PRETRAINED = True

    # Number of target species (approx 64.5k, exact count derived from data loader)
    NUM_CLASSES = 64500

    # ==========================================
    # Training Strategy (Two-Stage)
    # ==========================================

    # --- Stage 1: Representation Learning ---
    # Objective: Learn features using all data (Instance-Balanced Sampling)
    # Backbone + All Heads trainable
    STAGE1_EPOCHS = 4
    STAGE1_LR = 1e-3
    LABEL_SMOOTHING = 0.1

    # --- Stage 2: Classifier Re-balancing ---
    # Objective: Fine-tune Species Head for rare classes (Class-Balanced Sampling)
    # Backbone Frozen, Species Head Trainable
    STAGE2_EPOCHS = 1
    STAGE2_LR = 1e-4
