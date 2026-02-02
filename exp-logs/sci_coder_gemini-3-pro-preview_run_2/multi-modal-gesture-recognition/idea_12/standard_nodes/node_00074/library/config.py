import os
import torch


class Config:
    # =========================================================================
    # Global Paths & Settings
    # =========================================================================
    SEED = 42
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data (npy files)
    # Strategy: idea_13 corresponds to the current iteration
    WORK_DIR = "./working/idea_13"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # =========================================================================
    # Data & Feature Extraction
    # =========================================================================
    # 20 gestures + 1 background class (index 0)
    NUM_CLASSES = 21

    # Sampling and Sequence
    MAX_FRAMES = 300  # Fixed length for padding/truncating if batching requires it

    # Skeleton Features
    # We select the 12 Upper-Body Joints based on the dataset description order:
    # 1. HipCenter, 2. Spine, 3. ShoulderCenter, 4. Head,
    # 5. ShoulderLeft, 6. ElbowLeft, 7. WristLeft, 8. HandLeft,
    # 9. ShoulderRight, 10. ElbowRight, 11. WristRight, 12. HandRight
    # Indices are 0-based.
    SELECTED_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    NUM_JOINTS = len(SELECTED_JOINTS)

    # Input Channels:
    # 3 (Pos) + 3 (Vel) = 6 channels per joint
    # Total Skeleton Features = 12 * 6 = 72
    # Audio MFCCs = 13 (standard) or similar. Let's assume 13 MFCCs + 1 Energy = 14
    # Total Input Dim = 72 + 14 = 86 (Approximation, adjusted in model definition)
    INPUT_DIM = 72 + 13

    # Audio
    AUDIO_SR = 16000
    N_MFCC = 13

    # =========================================================================
    # Model Architecture: BA-MD-CRCN
    # =========================================================================
    # Stage 1: Recurrent Encoder
    LSTM_HIDDEN_DIM = 256
    LSTM_LAYERS = 2
    LSTM_DROPOUT = 0.3
    BIDIRECTIONAL = True

    # Stage 2 & 3: TCN (Temporal Convolutional Network)
    # Single-Stage TCN with dilated residuals
    TCN_NUM_CHANNELS = [256] * 10  # 10 layers
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.2

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15

    # Loss Weights
    # Ratio 0.1 Background : 1.0 Gesture
    BG_WEIGHT = 0.1
    GESTURE_WEIGHT = 1.0

    # Component Weights for Total Loss
    # L_total = L_cls + lambda_smooth * L_smooth
    LOSS_WEIGHT_CLS = 1.0
    LOSS_WEIGHT_SMOOTH = 1.0  # Unconditional smoothing penalty weight (Cite Lesson 68)

    # =========================================================================
    # Augmentation
    # =========================================================================
    # Sigma for Gaussian noise in position
    AUG_NOISE_SIGMA = 0.005
    # Sigma for smoothing kernel applied to noise (Low-pass filter)
    AUG_SMOOTH_SIGMA = 1.0

    @classmethod
    def init_dirs(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_class_weights(cls, device="cpu"):
        """
        Returns the class weights tensor for CrossEntropyLoss.
        Index 0 (Background) gets BG_WEIGHT.
        Indices 1-20 (Gestures) get GESTURE_WEIGHT.
        """
        weights = torch.ones(cls.NUM_CLASSES, device=device) * cls.GESTURE_WEIGHT
        weights[0] = cls.BG_WEIGHT
        return weights
