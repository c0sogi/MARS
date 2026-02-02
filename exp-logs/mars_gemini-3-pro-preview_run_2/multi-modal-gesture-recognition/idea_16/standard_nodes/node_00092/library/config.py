import os
import torch


class Config:
    """
    Configuration parameters for the Gated Latent-Transition Cascaded
    Recurrent-Convolutional Network (GLT-CRCN) pipeline.
    """

    # -------------------------------------------------------------------------
    # Paths and Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_16"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Data Processing Parameters
    # -------------------------------------------------------------------------
    # Skeleton Data
    # Indices for 12 Upper-Body Joints:
    # 0: HipCenter, 1: Spine, 2: ShoulderCenter, 3: Head
    # 4: ShoulderLeft, 5: ElbowLeft, 6: WristLeft, 7: HandLeft
    # 8: ShoulderRight, 9: ElbowRight, 10: WristRight, 11: HandRight
    SELECTED_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # Normalization
    REF_JOINT_INDEX = 0  # HipCenter for centering
    SCALE_FACTOR = 0.001  # Convert mm to meters

    # Audio Data
    AUDIO_SAMPLE_RATE = 16000
    N_MFCC = 13

    # Feature Dimensions
    # 12 joints * 3 coords (x,y,z) = 36
    # 12 joints * 3 coords (vx,vy,vz) = 36
    # Audio MFCCs = 13
    # Total Input Dim = 36 + 36 + 13 = 85
    INPUT_DIM = 85

    # -------------------------------------------------------------------------
    # Model Hyperparameters (GLT-CRCN)
    # -------------------------------------------------------------------------
    # Stage 1: Latent-Transition Recurrent Encoder (Bi-LSTM)
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2

    # Stage 2 & 3: Gated MS-TCN
    TCN_NUM_CHANNELS = 256
    TCN_NUM_LAYERS = 10  # Dilations: 2^0 to 2^9
    TCN_KERNEL_SIZE = 3
    DROPOUT = 0.5

    # Output
    NUM_CLASSES = 21  # 0: Background, 1-20: Gestures

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 8  # Cite Lesson 49: Larger batch size for stability
    NUM_EPOCHS = 40  # Cite Lesson 8: Sufficient epochs for convergence
    PATIENCE = 5  # Early stopping patience

    # Optimization
    LEARNING_RATE = 1e-3  # Cite Lesson 49: Higher LR for MS-TCN training
    WEIGHT_DECAY = 1e-4  # For AdamW

    # Loss Weights
    # Class weights: 0.1 for Background (0), 1.0 for Gestures (1-20)
    CLASS_WEIGHTS_LIST = [0.1] + [1.0] * 20

    # Unconditional Probability-Space Smoothing (T-MSE) Weight
    TMSE_WEIGHT = 0.15

    # -------------------------------------------------------------------------
    # Augmentation Parameters
    # -------------------------------------------------------------------------
    NOISE_STD = 0.01  # Standard deviation for Gaussian noise (in meters)
    TEMPORAL_FILTER_WIDTH = 5  # Width for low-pass filter on noise

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    @classmethod
    def get_class_weights(cls, device="cpu"):
        """Returns the class weights as a torch tensor on the specified device."""
        return torch.tensor(cls.CLASS_WEIGHTS_LIST, dtype=torch.float32).to(device)

    @classmethod
    def get_gesture_map(cls):
        """Returns the mapping of gesture names to IDs."""
        return {
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
