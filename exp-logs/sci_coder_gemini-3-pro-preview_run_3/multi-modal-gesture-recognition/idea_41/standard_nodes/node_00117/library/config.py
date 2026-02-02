import os


class Config:
    """
    Configuration for the Physically-Aligned Kinematic Refinement Network (PAK-RN).
    Centralizes all hyperparameters, file paths, and structural constraints.
    """

    # =========================================================================
    # 1. File System & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific experimental run (Idea 41)
    # Used for caching processed features and saving model checkpoints
    WORK_DIR = "./working/idea_41"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    MODEL_SAVE_PATH = os.path.join(WORK_DIR, "best_model.pth")

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # 2. Data Engineering & Physical Alignment
    # =========================================================================
    SEED = 42

    # Classes: 20 gestures + 1 background class (index 0)
    NUM_CLASSES = 21

    # Input Engineering Trilemma Resolution:
    # Deterministic Physical Scaling: Convert millimeters to meters.
    # This aligns position magnitude (~1.0) with normalized audio features
    # while preserving the signal-to-noise hierarchy (Pos >> Vel >> Acc).
    SKELETON_SCALE = 0.001

    # Audio configuration
    AUDIO_SAMPLE_RATE = 16000
    N_MFCC = 13

    # Sampling Strategy
    # Window size of 64 covers the typical gesture duration (~40-50 frames)
    # with context.
    WINDOW_SIZE = 64

    # Stride for training sample generation
    # Moderate stride (32) prevents overfitting while ensuring coverage.
    STRIDE_TRAIN = 32

    # =========================================================================
    # 3. Model Architecture (PAK-RN)
    # =========================================================================
    # Stage 1: Physically-Aligned Kinematic Encoder (Bi-GRU)
    # Hidden dimension: 96 units per direction => 192 total.
    # This avoids the bottleneck of 64-unit models and the instability of 128-unit models.
    GRU_HIDDEN_SIZE = 96
    GRU_LAYERS = 1

    # Stage 2 & 3: Monotonic Non-Causal Refinement (TCN)
    # Dilations increase monotonically to achieve a receptive field of 63 frames.
    TCN_DILATIONS = [1, 2, 4, 8, 16]
    TCN_KERNEL_SIZE = 3
    TCN_CHANNELS = 192  # Matches the Bi-GRU output dimension

    # Regularization
    DROPOUT = 0.2

    # =========================================================================
    # 4. Training Strategy
    # =========================================================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Training duration
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10

    # Loss Function Configuration
    # Weighted Cross-Entropy: Downweight the dominant background class
    LOSS_BG_WEIGHT = 0.2

    # Log-Space Smoothing Loss (Truncated MSE on log-probs)
    # Lambda = 0.15, Threshold = 1.0 (Low weight/threshold to prevent under-segmentation)
    SMOOTHING_LAMBDA = 0.15
    SMOOTHING_THRESHOLD = 1.0

    # =========================================================================
    # 5. Inference & Decoding
    # =========================================================================
    # Minimum duration in frames to consider a valid gesture prediction
    # Filters out spurious short spikes in probability.
    MIN_GESTURE_FRAMES = 5

    # Mapping for visualization and debugging
    # Maps internal class ID (1-20) to string name. Index 0 is background.
    LABEL_MAP = {
        1: "vattene",
        2: "vieniqui",
        3: "perfetto",
        4: "furbo",
        5: "cheduepalle",
        6: "chevuoi",
        7: "daccordo",
        8: "seipazzo",
        9: "combinato",
        10: "freganiente",
        11: "ok",
        12: "cosatifarei",
        13: "basta",
        14: "prendere",
        15: "noncenepiu",
        16: "fame",
        17: "tantotempo",
        18: "buonissimo",
        19: "messidaccordo",
        20: "sonostufo",
    }

    @staticmethod
    def ensure_dirs():
        """Creates necessary working directories."""
        os.makedirs(Config.WORK_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
