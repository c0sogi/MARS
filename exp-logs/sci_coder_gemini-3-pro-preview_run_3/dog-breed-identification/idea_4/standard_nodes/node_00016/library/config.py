import os
import torch


class Config:
    # ==========================================
    # 1. Environment & Reproducibility
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Matches available vCPUs

    # Debugging
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True

    # ==========================================
    # 2. Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files (Pre-generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Checkpoints and Cache
    WORKING_DIR = "./working/idea_4"
    OUTPUT_DIR = WORKING_DIR

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 3. Model Configuration
    # ==========================================
    MODEL_NAME = "convnext_base.fb_in1k"
    NUM_CLASSES = 120

    # ==========================================
    # 4. Training Strategy (Progressive Resolution)
    # ==========================================

    # General Optimization
    WEIGHT_DECAY = 1e-2
    LABEL_SMOOTHING = 0.0  # Strict CrossEntropy for Log Loss metric
    EARLY_STOPPING_PATIENCE = 5

    # --- Phase 1: Warmup ---
    # Goal: Align head weights while backbone is frozen
    PHASE1_EPOCHS = 1
    PHASE1_LR = 1e-3
    PHASE1_RES = 224
    PHASE1_BATCH_SIZE = 128

    # --- Phase 2: Standard Resolution Training ---
    # Goal: Learn structural features at standard resolution
    PHASE2_EPOCHS = 15
    PHASE2_LR = 5e-5
    PHASE2_RES = 224
    PHASE2_BATCH_SIZE = 32

    # --- Phase 3: High Resolution Fine-Tuning ---
    # Goal: Resolve fine-grained details (fur texture)
    PHASE3_EPOCHS = 15
    PHASE3_LR = 1e-5
    PHASE3_RES = 384
    PHASE3_BATCH_SIZE = 16  # Reduced batch size for larger resolution on A100

    # ==========================================
    # 5. Augmentation & Preprocessing
    # ==========================================
    # Augmentation parameters
    RAND_AUGMENT_N = 2
    RAND_AUGMENT_M = 9

    # Inference settings
    INFERENCE_RES = 384
    INFERENCE_BATCH_SIZE = 32
    TTA_FLIPS = True  # Enable Horizontal Flip TTA
