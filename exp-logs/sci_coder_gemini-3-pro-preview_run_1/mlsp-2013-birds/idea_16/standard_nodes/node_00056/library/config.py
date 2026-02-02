import os
import torch


class Config:
    """
    Central configuration for the Heterogeneous Ensemble Distillation experiment.
    """

    # =========================================================================
    # PATHS
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_16"

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)

    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Input Data Paths
    SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")

    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Submission Path
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # DATA CONFIGURATION
    # =========================================================================
    # Resolution: 256 (Height) x 640 (Width) to preserve frequency and temporal detail
    IMAGE_SIZE = (256, 640)

    NUM_CLASSES = 19

    # ImageNet Normalization Constants (Critical for SE-ResNet stability)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Mixup Augmentation
    MIXUP_ALPHA = 0.2

    # =========================================================================
    # MODEL CONFIGURATION
    # =========================================================================
    # Heterogeneous Teacher Ensemble: 2x ResNet34 + 2x SE-ResNet34
    TEACHER_MODELS = [
        {"arch": "resnet34", "id": "t_r34_0"},
        {"arch": "resnet34", "id": "t_r34_1"},
        {"arch": "legacy_seresnet34", "id": "t_se34_0"},
        {"arch": "legacy_seresnet34", "id": "t_se34_1"},
    ]

    # Student Model
    STUDENT_ARCH = "legacy_seresnet34"

    # =========================================================================
    # TRAINING HYPERPARAMETERS
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 32
    NUM_WORKERS = 4

    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Teacher Schedule
    TEACHER_EPOCHS = 50
    # SWA active for final 25% (approx last 12 epochs)
    TEACHER_SWA_START_EPOCH = 38
    TEACHER_SWA_LR = 1e-4

    # Student Schedule
    STUDENT_EPOCHS = 50
    # SWA active for final 30% (approx last 15 epochs)
    STUDENT_SWA_START_EPOCH = 35
    STUDENT_SWA_LR = 1e-4

    # =========================================================================
    # DEBUGGING
    # =========================================================================
    # Set DEBUG to True to run on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50

    @staticmethod
    def get_device():
        return torch.device(Config.DEVICE)
