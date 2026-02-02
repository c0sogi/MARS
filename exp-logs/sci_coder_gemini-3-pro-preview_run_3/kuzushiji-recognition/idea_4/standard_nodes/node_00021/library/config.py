import os
import torch


class Config:
    # ==========================================
    # 1. General & System Configuration
    # ==========================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # A100 has significant compute; 4 workers is a safe baseline for the 12 vCPUs
    NUM_WORKERS = 4

    # ==========================================
    # 2. Paths & Directories
    # ==========================================
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"

    # Ensure working subdirectories exist
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    OUTPUT_DIR = os.path.join(WORKING_DIR, "output")
    MODEL_DIR = os.path.join(WORKING_DIR, "models")

    for _dir in [WORKING_DIR, CACHE_DIR, OUTPUT_DIR, MODEL_DIR]:
        os.makedirs(_dir, exist_ok=True)

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Data Files
    UNICODE_MAP_PATH = os.path.join(INPUT_DIR, "unicode_translation.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Submission Output
    SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # Model Checkpoints & Artifacts
    DETECTOR_CHECKPOINT = os.path.join(MODEL_DIR, "detector_best.pth")
    CLASSIFIER_CHECKPOINT = os.path.join(MODEL_DIR, "classifier_best.pth")
    CLASS_MAP_PATH = os.path.join(WORKING_DIR, "class_map.npy")

    # ==========================================
    # 3. Data Processing Hyperparameters
    # ==========================================
    # Detector: Native resolution patches
    PATCH_SIZE = 1024
    DETECTOR_INPUT_SIZE = 1024
    DETECTOR_STRIDE = 4  # CenterNet standard output stride

    # Classifier: Crops of individual characters
    CLASSIFIER_INPUT_SIZE = 128

    # Normalization (Standard ImageNet)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ==========================================
    # 4. Model Architectures
    # ==========================================
    DETECTOR_BACKBONE = "resnet34"
    CLASSIFIER_BACKBONE = "resnet50"

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    # Debugging Flags
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200  # Small subset for rapid prototyping

    # Detector Training
    # Batch size 8 fits 1024x1024 patches on A100
    DETECTOR_BATCH_SIZE = 8
    DETECTOR_LR = 1e-4
    DETECTOR_EPOCHS = 30

    # Classifier Training
    # Batch size 128 for 128x128 crops
    CLASSIFIER_BATCH_SIZE = 128
    CLASSIFIER_LR = 1e-4
    CLASSIFIER_EPOCHS = 15

    # ==========================================
    # 6. Inference & Post-processing
    # ==========================================
    # Detection Thresholds
    SCORE_THRESHOLD = 0.3
    NMS_IOU_THRESHOLD = 0.2
    MAX_PREDICTIONS_PER_PAGE = 1200

    # Tiled Inference Settings
    TEST_TILE_SIZE = 1024
    # 25% overlap (1024 * 0.25 = 256 pixels overlap -> stride 768)
    TEST_TILE_STRIDE = 768
