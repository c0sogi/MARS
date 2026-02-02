import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # 1. Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # MegaDetector Results
    MEGADETECTOR_PATH = os.path.join(
        INPUT_DIR, "iwildcam2020_megadetector_results.json"
    )

    # Submission Output
    SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # --------------------------------------------------------------------------
    # 2. Model Configuration
    # --------------------------------------------------------------------------
    MODEL_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"  # timm model name
    NUM_CLASSES = 676  # Categories 0 to 675
    DROP_PATH_RATE = 0.2  # Stochastic depth rate
    DROPOUT_RATE = 0.2  # Classifier dropout

    # --------------------------------------------------------------------------
    # 3. Data Configuration
    # --------------------------------------------------------------------------
    IMAGE_SIZE = 448
    NUM_WORKERS = 8  # Optimized for 12 vCPUs

    # Augmentation Hyperparameters
    MIXUP_ALPHA = 0.8
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 0.5  # Probability of applying Mixup/Cutmix

    # --------------------------------------------------------------------------
    # 4. Training Configuration
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32
    NUM_EPOCHS = 12
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler
    WARMUP_EPOCHS = 1
    MIN_LR = 1e-6

    # --------------------------------------------------------------------------
    # 5. System & Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Caching
    # Path to save/load processed bounding box data
    CACHED_BBOXES_PATH = os.path.join(WORKING_DIR, "megadetector_boxes.parquet")

    @classmethod
    def print_config(cls):
        print("=" * 40)
        print("CONFIG")
        print("=" * 40)
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key}: {value}")
        print("=" * 40)
