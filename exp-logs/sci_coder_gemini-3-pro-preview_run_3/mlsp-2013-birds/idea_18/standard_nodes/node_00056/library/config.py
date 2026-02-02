import os
import torch


class Config:
    # ==========================================
    # General Settings
    # ==========================================
    PROJECT_NAME = "bird_classification_idea_18"
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea/experiment
    WORKING_DIR = "./working/idea_18"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Source Data Paths
    # We use the provided standard spectrograms as per the strategy
    SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")

    # ==========================================
    # Data & Preprocessing
    # ==========================================
    # Image Dimensions
    IMG_SIZE = (224, 224)

    # Input Channels: 3 (Replicating single channel to 3 for pretrained models)
    IN_CHANNELS = 3

    # Augmentation
    MIXUP_ALPHA = 0.4

    # ==========================================
    # Model Architecture
    # ==========================================
    # Heterogeneous Ensemble Backbones
    ARCHITECTURES = ["resnet18", "efficientnet_b0", "densenet121"]

    # Number of classes in the dataset
    NUM_CLASSES = 19

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    NUM_FOLDS = 5

    # Batch Size Strategy for Effective Batch Size of 64
    BATCH_SIZE = 16
    ACCUMULATION_STEPS = 4  # 16 * 4 = 64

    # Optimization
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # High regularization as requested

    # Early Stopping
    PATIENCE = 10

    # Checkpoint Strategy
    TOP_K_CHECKPOINTS = 3  # Save top 3 best models per fold for averaging

    # ==========================================
    # Evaluation / Inference
    # ==========================================
    # Test Time Augmentation Steps (Original, Shift Left, Shift Right)
    TTA_STEPS = 3

    @staticmethod
    def get_spectrogram_path(wav_filename):
        """
        Maps a WAV filename (e.g., 'PC10_... .wav') to its corresponding
        standard spectrogram path (e.g., '.../spectrograms/PC10_... .bmp').
        """
        base_name = os.path.splitext(os.path.basename(wav_filename))[0]
        return os.path.join(Config.SPECTROGRAM_DIR, f"{base_name}.bmp")
