import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching and intermediate files
    # Specific to "idea_35" as requested
    WORKING_DIR = "./working/idea_35"

    # Output directories
    SUBMISSION_DIR = "./submission"
    CHECKPOINT_DIR = "./checkpoints"

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Data Processing
    # -------------------------------------------------------------------------
    # Feature Selection: 12 Upper-Body Joints
    # Indices based on the provided dataset description:
    # 0: HipCenter, 1: Spine, 2: ShoulderCenter, 3: Head
    # 4: ShoulderLeft, 5: ElbowLeft, 6: WristLeft, 7: HandLeft
    # 8: ShoulderRight, 9: ElbowRight, 10: WristRight, 11: HandRight
    SELECTED_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # Input Dimensions
    # 12 joints * 3 coordinates (X, Y, Z) = 36 spatial features
    # + Audio MFCCs (typically 13 or 26, handled by dataset loader,
    # but we assume the model input dim adapts or is fixed sum)
    # Let's assume 12 joints * 3 = 36.
    # Audio dimension will be handled dynamically or set to standard 13 MFCCs.
    NUM_JOINTS = 12
    COORDS_PER_JOINT = 3
    AUDIO_FEATURE_DIM = 13
    INPUT_DIM = (NUM_JOINTS * COORDS_PER_JOINT) + AUDIO_FEATURE_DIM  # 49

    # Normalization
    SCALE_FACTOR = 0.001  # Convert mm to meters

    # -------------------------------------------------------------------------
    # Model Architecture: CRG-CN
    # -------------------------------------------------------------------------
    # Encoder
    CONV_STEM_LAYERS = 3
    CONV_STEM_KERNEL = 3
    CONV_STEM_STRIDE = 1
    CONV_STEM_PADDING = 1

    LSTM_HIDDEN_DIM = 256
    LSTM_NUM_LAYERS = 2
    LSTM_BIDIRECTIONAL = True

    # Refinement Stages (Gated MS-TCN)
    NUM_STAGES = 3  # 1 Encoder + 2 Refinement Stages
    TCN_NUM_LAYERS = 10
    TCN_CHANNELS = 256
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.5

    # Classes
    # 20 Gestures + 1 Background (Class 0 usually reserved for background or handled via index)
    # The dataset uses labels 1-20. We will map 1-20 to 1-20 and use 0 as background.
    NUM_CLASSES = 21

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 8  # A100 allows larger, but sequences are long
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP = 5.0

    # Early Stopping
    PATIENCE = 10

    # Loss Weights
    # Cross Entropy Weights: 0.1 for Background (index 0), 1.0 for Gestures
    CLASS_WEIGHTS = torch.ones(NUM_CLASSES)
    CLASS_WEIGHTS[0] = 0.1

    # Multi-Task Loss Components
    # L_total = L_cls + L_bnd + L_smooth
    LOSS_WEIGHT_CLS = 1.0
    LOSS_WEIGHT_BND = 1.0
    LOSS_WEIGHT_SMOOTH = 0.15  # T-MSE weight

    # -------------------------------------------------------------------------
    # Inference / Post-Processing
    # -------------------------------------------------------------------------
    MEDIAN_WINDOW_SIZE = 15  # For label smoothing

    # -------------------------------------------------------------------------
    # Debugging
    # -------------------------------------------------------------------------
    # Set to a small integer (e.g., 100) to limit dataset size during development
    # Set to None for full training
    DEBUG_SAMPLE_SIZE = None

    @classmethod
    def to_dict(cls):
        """Returns dictionary of configuration parameters for logging."""
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
