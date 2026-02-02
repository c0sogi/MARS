import os
import torch


class Config:
    # ==========================================
    #               Directories
    # ==========================================
    # Root directories
    INPUT_DIR = "./input"
    WORKING_DIR = "./working/idea_13"
    METADATA_DIR = "./metadata"

    # Specific data paths
    SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")

    # Metadata CSV paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Submission path
    SUBMISSION_PATH = "./submission/submission.csv"

    # ==========================================
    #            Global Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on vCPUs (12 available)

    # ==========================================
    #            Data Parameters
    # ==========================================
    # Image Dimensions
    # Height fixed at 256 to preserve frequency resolution
    # Width fixed at 640 for high-fidelity temporal morphology
    IMG_HEIGHT = 256
    IMG_WIDTH = 640

    # Normalization (ImageNet Stats) - Critical for pretrained weights
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    NUM_CLASSES = 19

    # ==========================================
    #          Training Hyperparameters
    # ==========================================
    # General
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Mixup
    MIXUP_ALPHA = 0.2

    # Epoch Schedules
    # Extended to 50 epochs to ensure convergence
    EPOCHS = 50

    # Stochastic Weight Averaging (SWA) Schedules
    # Teacher: Active for final 25% (Start ~Epoch 37)
    SWA_START_EPOCH_TEACHER = int(EPOCHS * 0.75)

    # Student: Active for final 30% (Start ~Epoch 35)
    SWA_START_EPOCH_STUDENT = int(EPOCHS * 0.70)

    # SWA Learning Rate
    SWA_LR = 5e-4

    # ==========================================
    #          Model Architecture
    # ==========================================
    # Using ResNet34
    BACKBONE_NAME = "resnet34"
    PRETRAINED = True

    @classmethod
    def setup(cls):
        """
        Ensures the working directory exists.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        # Ensure submission directory exists as well
        os.makedirs(os.path.dirname(cls.SUBMISSION_PATH), exist_ok=True)


# Run setup on import to guarantee directory existence
Config.setup()
