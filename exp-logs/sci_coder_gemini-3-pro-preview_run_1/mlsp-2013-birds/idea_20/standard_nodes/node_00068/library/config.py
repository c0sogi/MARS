import os
import torch


class Config:
    """
    Configuration for Heterogeneous Non-Attentive Ensemble Distillation with SWA.
    """

    # =========================================================================
    # Paths and Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_20"
    SUBMISSION_DIR = "./submission"

    # Ensure mutable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Data Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")

    # Output Paths
    TEACHER_CHECKPOINT_DIR = os.path.join(WORKING_DIR, "teachers")
    STUDENT_CHECKPOINT_DIR = os.path.join(WORKING_DIR, "student")
    PSEUDO_LABELS_PATH = os.path.join(WORKING_DIR, "pseudo_labels.parquet")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    os.makedirs(TEACHER_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(STUDENT_CHECKPOINT_DIR, exist_ok=True)

    # =========================================================================
    # Data Hyperparameters
    # =========================================================================
    SEED = 42
    NUM_CLASSES = 19

    # Image Preprocessing
    # High-Fidelity Alignment: 256 (H) x 640 (W)
    IMG_HEIGHT = 256
    IMG_WIDTH = 640
    IN_CHANNELS = 3  # Channel Replication (Grayscale -> RGB)

    # ImageNet Normalization
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # DataLoader
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Heterogeneous Ensemble: 2x ResNet-34 + 2x DenseNet-121
    # DenseNet-121 is chosen for architectural diversity without attention mechanisms
    TEACHER_ARCHS = ["resnet34", "resnet34", "densenet121", "densenet121"]
    STUDENT_ARCH = "resnet34"

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Mixup
    MIXUP_ALPHA = 0.2

    # Stochastic Weight Averaging (SWA)
    # Teachers: Active in final 25%
    SWA_START_EPOCH_TEACHER = int(EPOCHS * 0.75)
    # Student: Active in final 30%
    SWA_START_EPOCH_STUDENT = int(EPOCHS * 0.70)
    SWA_LR = 1e-4

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Helper Methods
    # =========================================================================
    @classmethod
    def get_teacher_path(cls, index, arch):
        """Returns the path for saving/loading a specific teacher model."""
        return os.path.join(
            cls.TEACHER_CHECKPOINT_DIR, f"teacher_{index}_{arch}_swa.pth"
        )

    @classmethod
    def get_student_path(cls):
        """Returns the path for saving/loading the student model."""
        return os.path.join(
            cls.STUDENT_CHECKPOINT_DIR, f"student_{cls.STUDENT_ARCH}_swa.pth"
        )
