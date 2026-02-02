import os


class GlobalConfig:
    """
    Global configuration for the Heterogeneous Resolution-Capacity Ensemble.
    Shared settings across all model streams.
    """

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_15"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- Reproducibility ---
    SEED = 42

    # --- Training Hyperparameters ---
    EPOCHS = 1000
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    OPTIMIZER = "Adam"

    # Scheduler: Cosine Annealing
    # Decoupled from stopping criteria to ensure full decay
    SCHEDULER = "CosineAnnealingLR"
    T_MAX = 1000

    # Loss Function: Mean Squared Error on dense signal
    LOSS_FUNCTION = "MSE"

    # --- Ensemble Strategy ---
    NUM_MODELS_PER_STREAM = 5
    TOTAL_MODELS = 10

    # Independent seeds for converged heterogeneous bagging
    STREAM_A_SEEDS = [42, 43, 44, 45, 46]
    STREAM_B_SEEDS = [47, 48, 49, 50, 51]

    # --- Data Processing ---
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # --- Inference ---
    # D4 Group TTA (8 views: Original, Rot90, Rot180, Rot270 + Flips)
    TTA_VIEWS = 8
    PAD_MODULUS = 16  # Ensure dimensions are multiples of 16 for U-Net alignment


class StreamAConfig:
    """
    Configuration for Stream A: Context Specialist.
    Standard 4-Level U-Net optimized for global structure.
    """

    NAME = "StreamA_Context"

    # Large patch size for global context
    PATCH_SIZE = 320

    # Architecture: 4-Level U-Net
    DEPTH = 4
    ENCODER_FILTERS = [32, 64, 128, 256]
    BOTTLENECK_FILTERS = 512

    # Standard bottleneck depth
    BOTTLENECK_DEPTH = 2

    # Upsampling strategy
    UPSAMPLING_MODE = "bilinear_conv"


class StreamBConfig:
    """
    Configuration for Stream B: Diversity Specialist.
    Deep-Bottleneck 3-Level U-Net optimized for local diversity and complex noise.
    """

    NAME = "StreamB_Diversity"

    # Smaller patch size for combinatorial crop diversity
    PATCH_SIZE = 160

    # Architecture: 3-Level U-Net
    DEPTH = 3
    ENCODER_FILTERS = [32, 64, 128]
    BOTTLENECK_FILTERS = 256

    # Deep High-Capacity Bottleneck (Innovation)
    # 8 consecutive Convolutional layers to model complex noise
    BOTTLENECK_DEPTH = 8

    # Upsampling strategy
    UPSAMPLING_MODE = "bilinear_conv"
