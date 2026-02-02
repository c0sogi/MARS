import os
import torch


class Config:
    """
    Configuration class for the Right Whale Detection task.
    Implements the settings for the Spectrally-Adaptive Hierarchical CA-ResNet-18 CRNN.
    """

    # ==========================================
    # General Configuration
    # ==========================================
    PROJECT_NAME = "idea_13"
    SEED = 42
    NUM_WORKERS = 4  # Adjust based on vCPU count (12 available)

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files (pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directories
    WORKING_DIR = f"./working/{PROJECT_NAME}"
    OUTPUT_DIR = WORKING_DIR  # Where checkpoints and cache are stored
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Audio Processing Parameters
    # ==========================================
    SR = 2000  # Sample Rate: 2kHz
    DURATION = 2.0  # Duration: 2.0 seconds
    N_SAMPLES = int(SR * DURATION)  # 4000 samples

    # Spectrogram Parameters
    N_MELS = 128  # Mel bins
    N_FFT = 256  # FFT window size (128ms at 2kHz)
    HOP_LENGTH = 20  # Hop length (10ms at 2kHz) -> ~200 time frames
    F_MIN = 15  # Min frequency
    F_MAX = 1000  # Max frequency (Nyquist)
    POWER = 2.0  # Power for Mel Spectrogram

    # Normalization
    NORM_MEAN = -4.2677  # Pre-calculated mean (approx)
    NORM_STD = 4.5689  # Pre-calculated std (approx)

    # ==========================================
    # Model Architecture Parameters
    # ==========================================
    MODEL_NAME = "SpectrallyAdaptiveHierarchicalCAResNet18CRNN"
    PRETRAINED = True
    IN_CHANNELS = 1  # Mono input
    NUM_CLASSES = 1

    # Backbone Constraints (ResNet Layers 2, 3, 4)
    # Layer 2: Standard stride (2, 2)
    # Layer 3 & 4: Asymmetric stride (2, 1) to preserve time
    RESNET_STRIDES = [(2, 2), (2, 1), (2, 1)]

    # Adaptive Spectral Pooling Bins for Hierarchical Fusion
    # Layer 2 -> 4 bins, Layer 3 -> 2 bins, Layer 4 -> 1 bin
    SPECTRAL_POOL_BINS = [4, 2, 1]

    # Bottleneck Fusion
    FUSION_CHANNELS = 512

    # Head
    RNN_HIDDEN_SIZE = 128
    RNN_LAYERS = 2
    DROPOUT = 0.3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 25
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Loss Function
    # Positive Class Weight to handle 1:9 imbalance
    POS_WEIGHT = 9.0

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 7

    # ==========================================
    # Augmentation Parameters
    # ==========================================
    # Mixup
    DO_MIXUP = True
    MIXUP_ALPHA = 0.4

    # SpecAugment
    # Time mask: Max 200ms. With hop=10ms, this is 20 frames.
    TIME_MASK_PARAM = 20
    FREQ_MASK_PARAM = 20

    # ==========================================
    # Utility Methods
    # ==========================================
    @staticmethod
    def get_pos_weight_tensor():
        """Returns the positive weight as a tensor for the loss function."""
        return torch.tensor([Config.POS_WEIGHT])

    @staticmethod
    def print_config():
        """Prints the current configuration."""
        print(f"--- Configuration: {Config.PROJECT_NAME} ---")
        print(f"SR: {Config.SR}, Duration: {Config.DURATION}s")
        print(f"Mels: {Config.N_MELS}, FFT: {Config.N_FFT}, Hop: {Config.HOP_LENGTH}")
        print(f"Model: {Config.MODEL_NAME}")
        print(f"Strides: {Config.RESNET_STRIDES}")
        print(f"Pool Bins: {Config.SPECTRAL_POOL_BINS}")
        print(f"Batch Size: {Config.BATCH_SIZE}, LR: {Config.LEARNING_RATE}")
        print(f"Mixup Alpha: {Config.MIXUP_ALPHA}, Pos Weight: {Config.POS_WEIGHT}")
        print(f"Working Dir: {Config.WORKING_DIR}")
        print("-------------------------------------------")
