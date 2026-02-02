import os
import torch


class Config:
    """
    Configuration class for the Attentive High-Fidelity SWA-Distillation pipeline.
    Centralizes all hyperparameters, file paths, and structural constants.
    """

    def __init__(self):
        # ==========================================
        # General Settings & Reproducibility
        # ==========================================
        self.SEED = 42
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.NUM_WORKERS = 4

        # ==========================================
        # Directories & Paths
        # ==========================================
        self.INPUT_DIR = "./input"
        self.METADATA_DIR = "./metadata"
        self.WORKING_DIR = "./working/idea_12"
        self.SUBMISSION_DIR = "./submission"

        # Ensure working directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

        self.SPECTROGRAM_DIR = os.path.join(
            self.INPUT_DIR, "supplemental_data", "spectrograms"
        )

        # Metadata paths
        self.TRAIN_METADATA_PATH = os.path.join(self.METADATA_DIR, "train.csv")
        self.VAL_METADATA_PATH = os.path.join(self.METADATA_DIR, "val.csv")
        self.TEST_METADATA_PATH = os.path.join(self.METADATA_DIR, "test.csv")

        # Output paths
        self.SUBMISSION_PATH = os.path.join(self.SUBMISSION_DIR, "submission.csv")
        self.TEACHER_CHECKPOINT_PREFIX = os.path.join(self.WORKING_DIR, "teacher_fold")
        self.STUDENT_CHECKPOINT_PATH = os.path.join(self.WORKING_DIR, "student_swa.pth")
        self.PSEUDO_LABELS_PATH = os.path.join(
            self.WORKING_DIR, "pseudo_labels.parquet"
        )

        # ==========================================
        # Data Parameters / High-Fidelity Input
        # ==========================================
        self.NUM_CLASSES = 19

        # High-Fidelity Resolution:
        # Height 256 (Frequency bins) x Width 640 (Time steps)
        self.IMG_HEIGHT = 256
        self.IMG_WIDTH = 640

        # Channel Replication (Mono -> RGB) to use pretrained weights
        self.CHANNELS = 3

        # ==========================================
        # Model Architecture
        # ==========================================
        # Using Squeeze-and-Excitation ResNet-34
        self.MODEL_NAME = "seresnet34"
        self.PRETRAINED = True

        # ==========================================
        # Training Hyperparameters
        # ==========================================
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 3e-4
        self.WEIGHT_DECAY = 1e-4

        # Augmentation
        self.MIXUP_ALPHA = 0.2
        self.PROB_CUTOUT = 0.5
        self.PROB_FLIP = 0.5

        # ==========================================
        # Stage 1: Teacher Ensemble (SWA)
        # ==========================================
        self.NUM_TEACHERS = 3
        self.TEACHER_EPOCHS = 40
        # Activate SWA for the final 25% of training
        self.TEACHER_SWA_START_EPOCH = 30
        self.TEACHER_SWA_LR = 1e-4

        # ==========================================
        # Stage 2: Pseudo-Labeling (TTA)
        # ==========================================
        # Deterministic TTA: Horizontal Flip
        self.USE_TTA = True

        # ==========================================
        # Stage 3: Student Training (SWA)
        # ==========================================
        self.STUDENT_EPOCHS = 50
        # Activate SWA for the final 30% of training
        self.STUDENT_SWA_START_EPOCH = 35
        self.STUDENT_SWA_LR = 1e-4

    def update(self, **kwargs):
        """
        Updates configuration parameters dynamically.
        Useful for debugging or adjusting epochs/batch_size via command line args.
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
                print(f"Config updated: {k} = {v}")
            else:
                print(f"Warning: Config has no attribute '{k}'")
