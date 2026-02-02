import os
import torch


class Config:
    """
    Configuration for the RGB-Temporal Dilated EfficientNet-B2 Speech Command Recognition Model.
    """

    # -------------------------------------------------------------------------
    # Reproducibility & Hardware
    # -------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 8  # Optimized for the 12 vCPU environment
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_ROOT = "./input"
    TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train", "audio")
    TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test", "audio")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output / Cache
    WORKING_DIR = "./working/idea_11"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Audio Signal Processing
    # -------------------------------------------------------------------------
    SAMPLE_RATE = 16000
    DURATION = 1.0  # seconds
    AUDIO_LEN = int(SAMPLE_RATE * DURATION)

    # Spectrogram Generation
    N_MELS = 128
    N_FFT = 1024  # ~64ms window size
    HOP_LENGTH = 160  # ~10ms hop size

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE = "tf_efficientnet_b2"
    PRETRAINED = True

    # Input Channels: 3
    # Channel 0: Log-Mel Spectrogram
    # Channel 1: Delta Features (Velocity)
    # Channel 2: Delta-Delta Features (Acceleration)
    IN_CHANNELS = 3

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 128  # Safe for A100 40GB
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Regularization
    MIXUP_ALPHA = 1.0  # High mixup for robust feature learning

    # -------------------------------------------------------------------------
    # Labels & Class Mapping
    # -------------------------------------------------------------------------
    # The 12 labels required for the submission
    TARGET_LABELS = {
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
    }
    SILENCE_LABEL = "silence"
    UNKNOWN_LABEL = "unknown"

    # The full set of fine-grained classes (Standard Google Speech Commands V2).
    # We train on these to preserve variance and learn distinct features for "unknown" words.
    FINE_GRAINED_CLASSES = sorted(
        [
            "backward",
            "bed",
            "bird",
            "cat",
            "dog",
            "down",
            "eight",
            "five",
            "follow",
            "forward",
            "four",
            "go",
            "happy",
            "house",
            "learn",
            "left",
            "marvin",
            "nine",
            "no",
            "off",
            "on",
            "one",
            "right",
            "seven",
            "sheila",
            "six",
            "stop",
            "three",
            "tree",
            "two",
            "up",
            "visual",
            "wow",
            "yes",
            "zero",
        ]
    )

    # The actual output classes of the model (Silence + 35 Words)
    CLASSES = [SILENCE_LABEL] + FINE_GRAINED_CLASSES
    NUM_CLASSES = len(CLASSES)

    @classmethod
    def get_class_index(cls, label):
        """
        Returns the integer index for a given fine-grained label.
        Used during dataset creation to encode labels.
        """
        try:
            return cls.CLASSES.index(label)
        except ValueError:
            # If a label in the dataset is not in our standard list,
            # we treat it as an auxiliary unknown word.
            # However, for training stability, we strictly expect known labels.
            # If the dataset contains a folder not in this list, it should be handled
            # by the dataloader (e.g. skipped or mapped to a generic bucket if we had one).
            # Here we raise to ensure data integrity.
            raise ValueError(
                f"Label '{label}' is not in the defined FINE_GRAINED_CLASSES list."
            )

    @classmethod
    def get_label_from_index(cls, index):
        """
        Returns the fine-grained string label for a given integer index.
        """
        return cls.CLASSES[index]

    @classmethod
    def map_fine_grained_to_target(cls, fine_grained_label):
        """
        Maps a fine-grained prediction (e.g., 'bed', 'yes', 'silence')
        to the final competition target label (e.g., 'unknown', 'yes', 'silence').
        """
        if fine_grained_label in cls.TARGET_LABELS:
            return fine_grained_label
        elif fine_grained_label == cls.SILENCE_LABEL:
            return cls.SILENCE_LABEL
        else:
            return cls.UNKNOWN_LABEL
