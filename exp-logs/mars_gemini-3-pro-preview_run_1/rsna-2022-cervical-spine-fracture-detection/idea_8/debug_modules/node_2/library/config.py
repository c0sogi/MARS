import os
import torch


class Config:
    # =========================================================================
    # Global Settings
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # Adjust based on vCPU count (12 available)

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Directories
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")
    SEGMENTATION_DIR = os.path.join(INPUT_DIR, "segmentations")

    # Metadata
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    TRAIN_BBOX_PATH = os.path.join(INPUT_DIR, "train_bounding_boxes.csv")

    # Output Directories (Idea 8)
    WORKING_DIR = "./working/idea_8"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Preprocessing
    # =========================================================================
    # DICOM Windowing (Bone Window)
    WINDOW_CENTER = 400
    WINDOW_WIDTH = 1800

    # Image Dimensions
    ORIGINAL_SIZE = 512
    CROP_SIZE = 256  # Size of the crop centered on the spine

    # Normalization (after windowing to 0-1)
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # =========================================================================
    # Stage 1: Multi-Class Anatomical Localizer (2D U-Net)
    # =========================================================================
    STAGE1_BACKBONE = "efficientnet-b0"
    STAGE1_IN_CHANNELS = 1  # Single slice CT
    STAGE1_NUM_CLASSES = 8  # Background + C1-C7

    STAGE1_BATCH_SIZE = 16
    STAGE1_EPOCHS = 15
    STAGE1_LR = 1e-4
    STAGE1_WEIGHT_DECAY = 1e-5

    # Paths for Stage 1 artifacts
    STAGE1_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "stage1_unet.pth")
    # Cache file for segmentation results (ROI coords, masks)
    STAGE1_CACHE_FILE = os.path.join(CACHE_DIR, "stage1_inference_results.parquet")

    # =========================================================================
    # Stage 2: Mask-Conditioned Detail Encoder (2.5D CNN)
    # =========================================================================
    STAGE2_BACKBONE = "tf_efficientnetv2_s"
    # Input: 3 slices (RGB) + 1 Bone Mask = 4 channels
    STAGE2_IN_CHANNELS = 4
    STAGE2_FEATURE_DIM = 1280  # Depends on backbone (effnetv2_s output)

    STAGE2_BATCH_SIZE = 32
    STAGE2_EPOCHS = 8
    STAGE2_LR = 3e-4
    STAGE2_WEIGHT_DECAY = 1e-4

    STAGE2_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "stage2_encoder.pth")
    # Cache file for extracted features
    STAGE2_TRAIN_FEATURES = os.path.join(CACHE_DIR, "stage2_train_features.npy")
    STAGE2_VAL_FEATURES = os.path.join(CACHE_DIR, "stage2_val_features.npy")
    STAGE2_TEST_FEATURES = os.path.join(CACHE_DIR, "stage2_test_features.npy")

    # =========================================================================
    # Stage 3: Hierarchical Anatomical Aggregator (Bi-GRU)
    # =========================================================================
    STAGE3_HIDDEN_DIM = 256
    STAGE3_NUM_LAYERS = 2
    STAGE3_DROPOUT = 0.3

    # Input dim = Visual Feature Dim + Anatomical Map Dim (7 classes)
    STAGE3_INPUT_DIM = STAGE2_FEATURE_DIM + 7

    STAGE3_BATCH_SIZE = 4  # Patient-level batch size (sequences are long)
    STAGE3_EPOCHS = 10
    STAGE3_LR = 5e-4
    STAGE3_WEIGHT_DECAY = 1e-4

    STAGE3_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "stage3_aggregator.pth")

    # Competition Metric Weights
    # patient_overall is weighted higher.
    # Weights roughly: patient_overall=1.0, others=average out.
    # Exact competition weights: w_j for fracture types.
    # We use a simplified weighting scheme for training:
    # C1-C7: 1.0, Patient: 7.0 (to balance the loss magnitude)
    LOSS_WEIGHTS = {
        "C1": 1.0,
        "C2": 1.0,
        "C3": 1.0,
        "C4": 1.0,
        "C5": 1.0,
        "C6": 1.0,
        "C7": 1.0,
        "patient_overall": 7.0,
    }
