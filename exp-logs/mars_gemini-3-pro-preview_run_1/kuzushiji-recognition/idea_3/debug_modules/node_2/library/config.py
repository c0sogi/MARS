import os
import torch
import pandas as pd


class Config:
    """
    Configuration for Kuzushiji Character Recognition.
    Implements the settings for Swin Transformer (Swin-B) + CenterNet strategy.
    """

    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 12  # Utilizing available vCPUs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Directories & Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORK_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Create necessary writable directories
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    UNICODE_MAP_PATH = os.path.join(INPUT_DIR, "unicode_translation.csv")

    # Image Directories
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test_images")

    # Output Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data & Preprocessing
    # -------------------------------------------------------------------------
    IMG_SIZE = 1024

    # ImageNet Normalization Statistics
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    # Swin-Base backbone (timm implementation name)
    BACKBONE = "swin_base_patch4_window7_224"

    # Channel dimensions for Swin-B at stages 1, 2, 3, 4
    # Used for FPN construction
    ENCODER_CHANNELS = [128, 256, 512, 1024]

    # Output channels for the Feature Pyramid Network
    FPN_OUT_CHANNELS = 256

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    NUM_EPOCHS = 40
    BATCH_SIZE = 4  # Conservative batch size for A100 40GB with Swin-B @ 1024x1024

    # Optimizer (AdamW) & Scheduler (Cosine) settings
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.05
    WARMUP_EPOCHS = 3
    MAX_GRAD_NORM = 1.0

    # CenterNet Loss Weights
    HM_LOSS_WEIGHT = 1.0  # Heatmap (Focal Loss)
    WH_LOSS_WEIGHT = 0.1  # Size Regression (L1)
    OFF_LOSS_WEIGHT = 1.0  # Local Offset Regression (L1)
    CLS_LOSS_WEIGHT = 1.0  # Classification at Center (Cross Entropy)

    # -------------------------------------------------------------------------
    # Augmentation (Geometric Only)
    # -------------------------------------------------------------------------
    # Strictly geometric augmentations to preserve stroke details
    AUG_SCALE_RANGE = (0.75, 1.25)  # Random scaling
    AUG_ROTATION = 15  # Random rotation (+/- degrees)
    AUG_TRANSLATE = 0.1  # Random shift (fraction of size)

    # -------------------------------------------------------------------------
    # Inference Configuration
    # -------------------------------------------------------------------------
    CONF_THRESHOLD = 0.1
    MAX_DETECTIONS = 1200  # Strict limit per page as per task description

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    @classmethod
    def get_class_mapping(cls):
        """
        Generates class mapping from the unicode translation file.

        Returns:
            char2id (dict): Mapping from Unicode label (e.g., 'U+306B') to Integer ID.
            id2char (dict): Mapping from Integer ID to Unicode label.
        """
        if not os.path.exists(cls.UNICODE_MAP_PATH):
            raise FileNotFoundError(f"Unicode map not found at {cls.UNICODE_MAP_PATH}")

        df = pd.read_csv(cls.UNICODE_MAP_PATH)

        # The file is expected to have a 'Unicode' column containing the labels
        if "Unicode" in df.columns:
            chars = df["Unicode"].values
        else:
            # Fallback: assume first column is the ID
            chars = df.iloc[:, 0].values

        # Create mappings based on the order in the file
        char2id = {c: i for i, c in enumerate(chars)}
        id2char = {i: c for i, c in enumerate(chars)}

        return char2id, id2char

    @classmethod
    def get_num_classes(cls):
        """Returns the total number of classes defined in the unicode map."""
        char2id, _ = cls.get_class_mapping()
        return len(char2id)
