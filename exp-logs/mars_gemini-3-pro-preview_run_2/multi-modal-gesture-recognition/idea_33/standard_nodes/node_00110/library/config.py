import os
import torch


class Config:
    """
    Configuration for the Feature-Injected Supervised Gated-Cascaded Network (FISG-CN).
    """

    # -------------------------------------------------------------------------
    # Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for the specific idea
    WORKING_DIR = "./working/idea_33"

    # Sub-directories for organization
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Ensure directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Data Preprocessing
    # -------------------------------------------------------------------------
    # Upper-body joints selection (indices based on Kinect skeleton format)
    # 1: HipCenter, 2: Spine, 3: ShoulderCenter, 4: Head
    # 5: ShoulderLeft, 6: ElbowLeft, 7: WristLeft, 8: HandLeft
    # 9: ShoulderRight, 10: ElbowRight, 11: WristRight, 12: HandRight
    # Note: Dataset indices might be 0-based or 1-based.
    # Assuming standard Kinect mapping where HipCenter is central.
    # We will use string names to be safe if the loader supports it,
    # otherwise we map based on the provided dataset description order.
    JOINTS_OF_INTEREST = [
        "HipCenter",
        "Spine",
        "ShoulderCenter",
        "Head",
        "ShoulderLeft",
        "ElbowLeft",
        "WristLeft",
        "HandLeft",
        "ShoulderRight",
        "ElbowRight",
        "WristRight",
        "HandRight",
    ]

    # Normalization
    SCALE_FACTOR = 0.001  # Convert millimeters to meters
    CENTER_JOINT = (
        "HipCenter"  # Joint to subtract from all others for translation invariance
    )

    # Audio
    USE_AUDIO = True
    AUDIO_MFCC_N_COEFFS = 13

    # Labels
    # 0 is background, 1-20 are gestures
    NUM_CLASSES = 21
    BACKGROUND_CLASS_ID = 0

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    # Input dimension calculation:
    # (12 joints * 3 coords) + (12 joints * 3 velocities) + 13 Audio MFCCs
    # = 36 + 36 + 13 = 85
    INPUT_DIM = 85

    HIDDEN_DIM = 256

    # Encoder (Stage 1)
    ENCODER_LAYERS = 2
    ENCODER_BIDIRECTIONAL = True
    ENCODER_DROPOUT = 0.3

    # Refinement (Stages 2 & 3)
    NUM_REFINEMENT_STAGES = 2
    TCN_LAYERS = 10
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.3

    # Feature Injection
    # If True, concatenates Encoder features to TCN inputs in refinement stages
    USE_FEATURE_INJECTION = True

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Early Stopping
    PATIENCE = 7

    # Loss Weights
    # Total Loss = W_cls * L_cls + W_bnd * L_bnd + W_smooth * L_smooth
    WEIGHT_CLS = 1.0
    WEIGHT_BND = 1.0
    WEIGHT_SMOOTH = 0.15  # T-MSE weight

    # Class Weights for Cross Entropy (Background vs Gestures)
    # Background gets lower weight (0.1) vs Gestures (1.0)
    BG_WEIGHT = 0.1
    GESTURE_WEIGHT = 1.0

    # Boundary Supervision
    BOUNDARY_SMOOTHING_WINDOW = 1  # +/- frames to label as boundary

    # -------------------------------------------------------------------------
    # Debugging
    # -------------------------------------------------------------------------
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50  # Number of samples to use if DEBUG is True

    @classmethod
    def get_class_weights_tensor(cls):
        """Returns the tensor for weighted CrossEntropyLoss."""
        weights = torch.ones(cls.NUM_CLASSES)
        weights[cls.BACKGROUND_CLASS_ID] = cls.BG_WEIGHT
        return weights.to(cls.DEVICE)

    def __str__(self):
        """Helper to print config."""
        return str(self.__class__.__dict__)
