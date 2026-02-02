import os
import torch


class Config:
    # ==========================================
    # 1. Global Constants & Hardware
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers

    # ==========================================
    # 2. Paths
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test_images")
    UNICODE_MAP_PATH = os.path.join(INPUT_DIR, "unicode_translation.csv")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    WORKING_DIR = "./working/idea_3"
    OUTPUT_DIR = os.path.join(WORKING_DIR, "output")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Path to save/load the class mapping (Unicode <-> ID)
    CLASS_MAP_PATH = os.path.join(WORKING_DIR, "class_map.npy")

    # ==========================================
    # 3. Data Configuration
    # ==========================================
    # Normalization stats from Data Analysis
    NORM_MEAN = [0.7470, 0.7053, 0.6315]
    NORM_STD = [0.2186, 0.2123, 0.2005]

    # ==========================================
    # 4. Stage 1: Detector (HRNet-W32) Config
    # ==========================================
    DETECTOR_MODEL_NAME = "hrnet_w32"
    DETECTOR_INPUT_SIZE = (1024, 1024)  # Height, Width (Square crops)
    DETECTOR_OUTPUT_STRIDE = 4  # HRNet/CenterNet standard stride
    DETECTOR_NUM_CLASSES = 1  # Class-agnostic detection (Character vs Background)

    DETECTOR_BATCH_SIZE = 8  # Fits in A100 40GB with HRNet
    DETECTOR_LR = 1e-4
    DETECTOR_EPOCHS = 20

    # Loss Weights
    LOSS_HEATMAP_WEIGHT = 1.0
    LOSS_SIZE_WEIGHT = 0.1
    LOSS_OFFSET_WEIGHT = 1.0

    # Gaussian Radius calculation for Heatmap
    GAUSSIAN_IOU = 0.7

    # ==========================================
    # 5. Stage 2: Classifier (ResNet-34) Config
    # ==========================================
    CLASSIFIER_MODEL_NAME = "resnet34"
    CLASSIFIER_INPUT_SIZE = (64, 64)  # Crops resized to this
    CLASSIFIER_BATCH_SIZE = 256
    CLASSIFIER_LR = 1e-3
    CLASSIFIER_EPOCHS = 10

    # ==========================================
    # 6. Augmentation Config
    # ==========================================
    SCALE_RANGE = (0.7, 1.3)  # Random scaling for detector
    ROTATION_RANGE = 5  # Degrees (+/-)

    # ==========================================
    # 7. Inference / Tiling Config
    # ==========================================
    TILE_SIZE = 1024
    TILE_OVERLAP = 0.25  # 25% overlap between tiles

    SCORE_THRESHOLD = 0.3  # Minimum heatmap score to consider a detection
    NMS_IOU_THRESHOLD = 0.2  # Global NMS threshold to merge tile predictions
    MAX_DETECTIONS_PER_PAGE = 1200  # Constraint from task description

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        print(f"Configuration setup complete. Working directory: {cls.WORKING_DIR}")
