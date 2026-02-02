import os
import torch


class Config:
    """
    Configuration class for Idea 11: High-Fidelity Aligned Distillation with
    Robust Augmentation and Dual-Stage SWA.
    """

    def __init__(self, debug=False):
        """
        Initialize configuration.

        Args:
            debug (bool): If True, sets up a lightweight configuration for debugging
                          (fewer epochs, subset of data).
        """
        # --- System & Reproducibility ---
        self.SEED = 42
        self.NUM_WORKERS = 4
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Paths ---
        self.INPUT_DIR = "./input"
        self.METADATA_DIR = "./metadata"
        self.WORKING_DIR = "./working/idea_11"

        # Ensure working directory exists
        os.makedirs(self.WORKING_DIR, exist_ok=True)

        self.SPECTROGRAM_DIR = os.path.join(
            self.INPUT_DIR, "supplemental_data", "spectrograms"
        )
        self.TRAIN_CSV = os.path.join(self.METADATA_DIR, "train.csv")
        self.VAL_CSV = os.path.join(self.METADATA_DIR, "val.csv")
        self.TEST_CSV = os.path.join(self.METADATA_DIR, "test.csv")
        self.SUBMISSION_PATH = "./submission/submission.csv"

        # --- Data Processing ---
        # High-Fidelity Resolution: 256px height (native freq) x 640px width
        self.IMG_HEIGHT = 256
        self.IMG_WIDTH = 640
        self.CHANNELS = 3  # Replicated channels for ImageNet weights

        # Debugging: Limit dataset size if debug is True
        self.MAX_SAMPLES = 50 if debug else None

        # --- Model Architecture ---
        self.BACKBONE = "resnet34"
        self.NUM_CLASSES = 19
        self.PRETRAINED = True

        # --- Optimization ---
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 3e-4
        self.WEIGHT_DECAY = 1e-4

        # --- Augmentation Strategy ---
        # Robust regularization to support high resolution
        self.MIXUP_ALPHA = 0.2
        self.USE_HORIZONTAL_FLIP = True  # Re-introduced to prevent regression
        self.USE_CUTOUT = True  # Unstructured masking

        # --- Teacher Training (Stage 1) ---
        self.NUM_TEACHERS = 3
        if debug:
            self.TEACHER_EPOCHS = 2
            self.TEACHER_SWA_START_EPOCH = 1
        else:
            self.TEACHER_EPOCHS = 40
            # SWA active for final 25%
            self.TEACHER_SWA_START_EPOCH = 30

        # --- Pseudo Labeling (Stage 2) ---
        self.PSEUDO_LABEL_PATH = os.path.join(self.WORKING_DIR, "pseudo_labels.parquet")

        # --- Student Training (Stage 3) ---
        if debug:
            self.STUDENT_EPOCHS = 2
            self.STUDENT_SWA_START_EPOCH = 1
        else:
            self.STUDENT_EPOCHS = 50
            # SWA active for final 30%
            self.STUDENT_SWA_START_EPOCH = 35

        # --- Checkpoints ---
        self.TEACHER_CHECKPOINT_TEMPLATE = os.path.join(
            self.WORKING_DIR, "teacher_{}_swa.pth"
        )
        self.STUDENT_CHECKPOINT = os.path.join(self.WORKING_DIR, "student_swa.pth")

    def __repr__(self):
        return (
            f"Config(debug={self.MAX_SAMPLES is not None}, "
            f"size={self.IMG_HEIGHT}x{self.IMG_WIDTH}, "
            f"teacher_epochs={self.TEACHER_EPOCHS}, "
            f"student_epochs={self.STUDENT_EPOCHS})"
        )
