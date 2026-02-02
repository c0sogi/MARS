import os
import torch


class Config:
    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for specific experiment idea
    WORKING_DIR = "./working/idea_8"

    # Cache directory for processed features (npy files)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Checkpoint directory for saving models
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # Data Processing
    # ==========================================
    # Gesture Vocabulary
    LABEL_MAP = {
        "vattene": 1,
        "vieniqui": 2,
        "perfetto": 3,
        "furbo": 4,
        "cheduepalle": 5,
        "chevuoi": 6,
        "daccordo": 7,
        "seipazzo": 8,
        "combinato": 9,
        "freganiente": 10,
        "ok": 11,
        "cosatifarei": 12,
        "basta": 13,
        "prendere": 14,
        "noncenepiu": 15,
        "fame": 16,
        "tantotempo": 17,
        "buonissimo": 18,
        "messidaccordo": 19,
        "sonostufo": 20,
    }
    # Inverse map for decoding
    ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}

    # 20 gestures + 1 background class (index 0)
    NUM_CLASSES = 21

    # Audio Feature Extraction
    AUDIO_SAMPLE_RATE = 16000
    N_MFCC = 13

    # Skeleton Features
    # 20 joints * 3 coordinates (X, Y, Z)
    NUM_JOINTS = 20
    COORDS_PER_JOINT = 3
    SKELETON_INPUT_DIM = NUM_JOINTS * COORDS_PER_JOINT  # 60

    # ==========================================
    # Model Architecture (KA-MTRN)
    # ==========================================
    # Tri-Stream Input Dimensions
    POSE_INPUT_DIM = SKELETON_INPUT_DIM  # 60
    VELOCITY_INPUT_DIM = SKELETON_INPUT_DIM  # 60
    AUDIO_INPUT_DIM = N_MFCC  # 13

    # Convolutional Stem
    CNN_KERNEL_SIZE = 7
    CNN_STRIDE = 1
    CNN_PADDING = 3  # (k-1)/2 to maintain length

    # Recurrent Backbone
    HIDDEN_DIM = 256
    NUM_RNN_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 8  # Micro-batching for small dataset
    NUM_EPOCHS = 50  # Maximum epochs
    LEARNING_RATE = 1e-3  # Initial LR for AdamW
    WEIGHT_DECAY = 0.05  # Aggressive regularization

    # Loss Configuration
    LABEL_SMOOTHING = 0.1
    BACKGROUND_WEIGHT = 0.5  # Weight for class 0 in CrossEntropy
    BOUNDARY_LOSS_WEIGHT = 0.5

    # Optimization
    EARLY_STOPPING_PATIENCE = 10
    GRADIENT_CLIP_VAL = 1.0

    # ==========================================
    # Inference / Post-Processing
    # ==========================================
    MEDIAN_FILTER_KERNEL = 5
    MIN_SEGMENT_LENGTH = 5

    @classmethod
    def setup(cls):
        """Creates necessary directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_device(cls):
        """Returns the appropriate torch device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


# Initialize directories on import
Config.setup()
