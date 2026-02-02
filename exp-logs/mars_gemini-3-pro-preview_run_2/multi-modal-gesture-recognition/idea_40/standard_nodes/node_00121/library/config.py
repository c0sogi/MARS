import os
import torch


class Config:
    """
    Configuration for Geometric Multi-Granularity Convolutional-Recurrent Gated Network (GMG-CRGN).
    Acts as the single source of truth for hyperparameters, paths, and model settings.
    """

    # -------------------------------------------------------------------------
    # Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea (Idea 40)
    WORK_DIR = "./working/idea_40"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORK_DIR, "submission")

    # Ensure working directories exist immediately upon import
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Global Settings
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For data loading

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    # 20 Gestures + 1 Background class (Index 0)
    NUM_CLASSES = 21

    # Skeleton: 12 Upper-Body Joints
    # Indices based on the provided dataset description (Kinect format)
    # 0: HipCenter, 1: Spine, 2: ShoulderCenter, 3: Head
    # 4: ShoulderLeft, 5: ElbowLeft, 6: WristLeft, 7: HandLeft
    # 8: ShoulderRight, 9: ElbowRight, 10: WristRight, 11: HandRight
    SELECTED_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # Reference joint for centering (HipCenter)
    REF_JOINT_INDEX = 0

    # Bone Connections for Vector Calculation (Parent, Child)
    # Indices correspond to the original skeleton indices
    BONE_PAIRS = [
        (0, 1),  # HipCenter -> Spine
        (1, 2),  # Spine -> ShoulderCenter
        (2, 3),  # ShoulderCenter -> Head
        (2, 4),  # ShoulderCenter -> ShoulderLeft
        (4, 5),  # ShoulderLeft -> ElbowLeft
        (5, 6),  # ElbowLeft -> WristLeft
        (6, 7),  # WristLeft -> HandLeft
        (2, 8),  # ShoulderCenter -> ShoulderRight
        (8, 9),  # ShoulderRight -> ElbowRight
        (9, 10),  # ElbowRight -> WristRight
        (10, 11),  # WristRight -> HandRight
    ]

    # Normalization
    SCALE_FACTOR = 0.001  # Convert millimeters to meters

    # Audio Features
    AUDIO_SR = 16000
    AUDIO_N_MFCC = 13
    AUDIO_N_FFT = 2048
    AUDIO_HOP_LENGTH = 512

    # -------------------------------------------------------------------------
    # Model Architecture (GMG-CRGN)
    # -------------------------------------------------------------------------
    # Input Dimension Calculation:
    # Joints: 12 joints * (3 Pos + 3 Vel) = 72
    # Bones: 11 bones * 3 = 33
    # Audio: 13 MFCCs
    # Total Input Dim = 72 + 33 + 13 = 118
    INPUT_DIM = 118

    # Multi-Granularity Stem
    STEM_KERNEL_SIZES = [3, 7, 11]

    # Backbone (Bi-LSTM)
    HIDDEN_DIM = 256
    LSTM_LAYERS = 2
    LSTM_BIDIRECTIONAL = True

    # MS-TCN (Refinement Stages)
    MSTCN_STAGES = 2  # Number of refinement stages (Stage 2 and 3)
    MSTCN_LAYERS = 10  # Layers per stage (dilations 1, 2, ..., 512)
    MSTCN_CHANNELS = 256
    MSTCN_DROPOUT = 0.5

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    EPOCHS = 50
    BATCH_SIZE = 8
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 5.0

    # Early Stopping
    PATIENCE = 10

    # Loss Weights
    # Class weights: 0.1 for background (0), 1.0 for gestures (1-20)
    CLASS_WEIGHT_BG = 0.1
    CLASS_WEIGHT_GESTURE = 1.0

    # Multi-Task Loss Components
    LAMBDA_CLS = 1.0  # Classification
    LAMBDA_BND = 1.0  # Boundary (Focal Loss)
    LAMBDA_SMOOTH = 0.15  # T-MSE Smoothing

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50  # Only use 50 samples if DEBUG is True

    @staticmethod
    def get_class_weights():
        """Returns the weight tensor for CrossEntropyLoss."""
        weights = [Config.CLASS_WEIGHT_BG] + [Config.CLASS_WEIGHT_GESTURE] * (
            Config.NUM_CLASSES - 1
        )
        return torch.tensor(weights, dtype=torch.float32)
