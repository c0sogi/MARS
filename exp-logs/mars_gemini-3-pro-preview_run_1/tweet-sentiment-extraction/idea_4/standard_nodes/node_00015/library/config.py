import os
import torch


class Config:
    # --- General Settings ---
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Paths ---
    # Metadata files generated in the previous step
    TRAIN_META_PATH = "./metadata/train_metadata.csv"
    VAL_META_PATH = "./metadata/validation_metadata.csv"
    TEST_META_PATH = "./metadata/test_metadata.csv"

    # Output paths
    WORKING_DIR = "./working/idea_4/"
    SUBMISSION_DIR = "./submission/"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- Model Architecture ---
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LEN = 128
    HIDDEN_SIZE = 768
    # Weighted Layer Pooling settings
    N_POOLING_LAYERS = 4
    # Multi-Task Head settings
    USE_AUX_HEAD = True

    # --- Training Hyperparameters ---
    EPOCHS = 5
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32

    # Optimization
    LEARNING_RATE = 5e-5  # Corrected base LR
    WEIGHT_DECAY = 0.01
    LLRD_DECAY = 0.9  # Layer-wise Learning Rate Decay
    eps = 1e-6
    betas = (0.9, 0.999)
    MAX_GRAD_NORM = 1000

    # Scheduler
    SCHEDULER_TYPE = "cosine"
    NUM_WARMUP_STEPS = 0  # 0 or small percentage of total steps

    # --- Advanced Techniques ---
    # Data Processing
    FILTER_NEUTRAL_TRAIN = True  # Exclude neutral tweets from training
    TARGET_SMOOTHING_SIGMA = 1.0  # For Gaussian-smoothed start/end targets

    # Loss Weights
    AUX_LOSS_WEIGHT = 1.0  # Weight for the dense mask auxiliary task

    # Adversarial Weight Perturbation (AWP)
    USE_AWP = True
    AWP_START_EPOCH = 2  # Start AWP after model has stabilized
    AWP_LR = 1e-4
    AWP_EPS = 1e-2

    # Early Stopping
    PATIENCE = 3

    # --- Debugging ---
    # Set to a small integer (e.g., 100) to run on a subset of data for testing pipeline
    # Set to None for full training
    DEBUG_SAMPLE_SIZE = None

    def __str__(self):
        """Helper to print config settings."""
        attributes = [
            a
            for a in dir(self)
            if not a.startswith("__") and not callable(getattr(self, a))
        ]
        return "\n".join([f"{a}: {getattr(self, a)}" for a in attributes])
