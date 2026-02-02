import os
import torch


class Config:
    """
    Configuration for Idea 6: Swin Transformer-based Cascade R-CNN.
    Centralizes all hyperparameters for data, model, training, and inference.
    """

    # =======================
    # General Settings
    # =======================
    SEED = 42
    DEBUG = False  # Default flag, can be overridden by setup()
    DEBUG_SAMPLE_SIZE = 100  # Number of images to use when DEBUG is True

    # =======================
    # Directories & Paths
    # =======================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for Idea 6
    WORKING_DIR = "./working/idea_6"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Cache directory for deterministic data processing
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Metadata Paths (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Model Checkpoint & Submission Paths
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # =======================
    # Data Preprocessing
    # =======================
    # Letterbox resizing target size (longest dimension)
    IMG_SIZE = 640

    # Normalization constants (ImageNet defaults)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # =======================
    # Augmentation Parameters
    # =======================
    # Strategy: RandomRotate90, ShiftScaleRotate, Cutout (No Mosaic)
    AUG_ROTATE_90 = True
    AUG_SHIFT_SCALE_ROTATE_PROB = 0.5
    AUG_ROTATE_LIMIT = 30
    AUG_SCALE_LIMIT = 0.1
    AUG_SHIFT_LIMIT = 0.1

    AUG_CUTOUT_PROB = 0.5
    AUG_CUTOUT_NUM_HOLES = 8
    AUG_CUTOUT_MAX_H_SIZE = 32
    AUG_CUTOUT_MAX_W_SIZE = 32

    # =======================
    # Model Architecture
    # =======================
    # Backbone: Swin Transformer Base (using timm)
    BACKBONE_NAME = "swin_base_patch4_window7_224"
    # Output channels for Swin Base at stages 1, 2, 3, 4
    BACKBONE_OUT_CHANNELS = [128, 256, 512, 1024]

    # Feature Pyramid Network (FPN) out channels
    FPN_OUT_CHANNELS = 256

    # Detection Head (Cascade R-CNN)
    # Class 0: Background, Class 1: Opacity
    NUM_CLASSES_DETECTION = 2
    DETECTION_CLASSES = ["background", "opacity"]

    # Anchor Generator Settings
    RPN_ANCHOR_SCALES = [32, 64, 128, 256, 512]
    RPN_ANCHOR_RATIOS = [0.5, 1.0, 2.0]

    # Cascade R-CNN IoU Thresholds
    CASCADE_IOU_THRESHOLDS = [0.5, 0.6, 0.7]

    # Study Classification Head
    # Classes: Negative, Typical, Indeterminate, Atypical
    NUM_CLASSES_STUDY = 4
    STUDY_CLASSES = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]

    # =======================
    # Training Hyperparameters
    # =======================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 4  # Adjusted for A100 + Swin Base memory usage
    NUM_WORKERS = 4

    EPOCHS = 12
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.05
    CLIP_GRAD_NORM = 10.0

    # Optimizer & Scheduler
    OPTIMIZER = "AdamW"
    SCHEDULER = "CosineAnnealingLR"
    WARMUP_EPOCHS = 1
    MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 4
    EARLY_STOPPING_MIN_DELTA = 1e-4

    # Loss Weights (Multi-task learning)
    LOSS_WEIGHT_RPN_CLS = 1.0
    LOSS_WEIGHT_RPN_BOX = 1.0
    LOSS_WEIGHT_ROI_CLS = 1.0
    LOSS_WEIGHT_ROI_BOX = 1.0
    LOSS_WEIGHT_STUDY = 2.0  # Emphasize study classification

    # Focal Loss Settings (for Study Classification)
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # =======================
    # Inference Settings
    # =======================
    # Confidence threshold to consider a detection valid
    CONF_THRESHOLD = 0.001
    # NMS threshold for final predictions
    NMS_IOU_THRESHOLD = 0.5

    # Submission format constants
    NONE_CLASS_ID = "none"
    OPACITY_CLASS_ID = "opacity"

    @classmethod
    def setup(cls, debug=False, epochs=None, batch_size=None):
        """
        Configure the environment and hyperparameters.

        Args:
            debug (bool): If True, enables debug mode (fewer data samples, fewer epochs).
            epochs (int, optional): Override default number of epochs.
            batch_size (int, optional): Override default batch size.
        """
        cls.DEBUG = debug

        if epochs is not None:
            cls.EPOCHS = epochs

        if batch_size is not None:
            cls.BATCH_SIZE = batch_size

        if cls.DEBUG:
            print(f"[Config] Debug mode ENABLED.")
            cls.EPOCHS = 2 if epochs is None else epochs
            cls.BATCH_SIZE = 2 if batch_size is None else batch_size
            cls.NUM_WORKERS = 0  # Avoid multiprocessing overhead in debug
            print(
                f"[Config] Overrides: Epochs={cls.EPOCHS}, BatchSize={cls.BATCH_SIZE}, Samples={cls.DEBUG_SAMPLE_SIZE}"
            )
        else:
            print(
                f"[Config] Running in FULL mode. Epochs={cls.EPOCHS}, BatchSize={cls.BATCH_SIZE}"
            )
