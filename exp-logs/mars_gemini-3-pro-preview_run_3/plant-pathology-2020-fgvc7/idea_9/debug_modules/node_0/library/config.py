import os
import torch


class Config:
    """
    Global configuration for the Apple Disease Detection task.
    Implements the 'Heterogeneous Ensemble of Multi-Axis and Convolutional Experts' strategy.
    """

    # ==== General Settings ====
    SEED = 42
    debug = False  # Set to True for quick debugging
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode

    # ==== Directories ====
    # Read-only input directories
    INPUT_DIR = "./input"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Metadata directory (pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output directory for this specific idea/experiment
    # We ensure this directory exists
    WORK_DIR = "./working/idea_9"
    os.makedirs(WORK_DIR, exist_ok=True)

    SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # ==== Data ====
    NUM_CLASSES = 4
    CLASS_LABELS = ["healthy", "multiple_diseases", "rust", "scab"]

    # ==== Model Architecture ====
    # Backbone 1: EfficientNet-B4 (Texture Expert)
    EFFNET_MODEL_NAME = "tf_efficientnet_b4"
    EFFNET_IMG_SIZE = 380

    # Backbone 2: MaxViT-Tiny (Global-Local Hybrid Expert)
    # Using timm's naming convention. .in1k implies ImageNet-1k pretrained
    MAXVIT_MODEL_NAME = "maxvit_tiny_tf_224.in1k"
    MAXVIT_IMG_SIZE = 224

    # ==== Training Hyperparameters ====
    NUM_FOLDS = 5
    EPOCHS = 25
    PATIENCE = 10  # Relaxed early stopping

    # Batch size: Adjusted for A100 40GB.
    # MaxViT can be memory intensive, so we use a conservative batch size.
    BATCH_SIZE = 16

    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 10.0

    # Optimizer & Scheduler
    SCHEDULER_T_0 = 25  # Cosine annealing cycle length (matches epochs)
    SCHEDULER_MIN_LR = 1e-6

    # ==== Augmentation ====
    # Strong geometric augmentations
    AUG_SHIFT_LIMIT = 0.2
    AUG_SCALE_LIMIT = 0.2
    AUG_ROTATE_LIMIT = 20

    # ==== Inference ====
    # Test Time Augmentation (TTA)
    TTA_FLIP_HORIZONTAL = True
    TTA_FLIP_VERTICAL = False  # Explicitly excluded per strategy
    TTA_TRANSPOSE = False  # Explicitly excluded per strategy

    # ==== Hardware ====
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = (
        4  # 12 vCPUs available, 4 is usually a safe sweet spot for data loading
    )

    @classmethod
    def get_image_size(cls, model_type):
        """
        Returns the appropriate image size based on the model type.
        Args:
            model_type (str): 'effnet' or 'maxvit'
        """
        if model_type == "effnet":
            return cls.EFFNET_IMG_SIZE
        elif model_type == "maxvit":
            return cls.MAXVIT_IMG_SIZE
        else:
            raise ValueError(f"Unknown model type: {model_type}")
