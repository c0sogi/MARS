import os
import torch


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    # Input Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    ESSENTIAL_DATA_DIR = os.path.join(INPUT_DIR, "essential_data")
    SUPPLEMENTAL_DATA_DIR = os.path.join(INPUT_DIR, "supplemental_data")

    # Specific Data Paths
    SPECTROGRAM_DIR = os.path.join(SUPPLEMENTAL_DATA_DIR, "spectrograms")
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    WORKING_DIR = "./working/idea_8"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Configuration
    # ==========================================
    MODEL_NAME = "resnet34"
    NUM_CLASSES = 19
    PRETRAINED = True
    DROPOUT = 0.0  # ResNet usually doesn't need high dropout in the head

    # ==========================================
    # Data Preprocessing
    # ==========================================
    # "Densified Global Resize": 256 (freq) x 512 (time)
    IMG_HEIGHT = 256
    IMG_WIDTH = 512
    CHANNELS = 3  # Channel replication (Gray -> RGB)

    # Normalization (ImageNet stats)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4

    # Mixup
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2

    # ==========================================
    # Pipeline: Ensemble & Distillation
    # ==========================================
    # Stage 1: Teacher Training
    NUM_TEACHERS = 3
    TEACHER_EPOCHS = 35  # Increased for Mixup convergence

    # Stage 3: Student Training with SWA
    # SWA applied in the final ~25% of epochs
    STUDENT_EPOCHS = 50
    SWA_START_EPOCH = 35
    SWA_LR = 1e-4  # Constant LR for SWA phase

    # ==========================================
    # System & Debugging
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging flags to control dataset size
    DEBUG = False
    DEBUG_SUBSET_SIZE = 20

    @classmethod
    def setup(cls):
        """
        Initialize necessary directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon import
Config.setup()
