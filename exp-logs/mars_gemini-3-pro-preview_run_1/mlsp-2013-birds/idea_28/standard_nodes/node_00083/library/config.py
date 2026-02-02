import os


class Config:
    # ==========================================
    # Paths and Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_28"
    SUBMISSION_DIR = "./submission"

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Specific Data Paths
    SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache path for processed datasets (if needed)
    CACHE_DIR = WORKING_DIR

    # ==========================================
    # Global Configuration
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4  # Optimized for 12 vCPUs
    DEVICE = "cuda"  # Assumes GPU availability

    # Debugging / Development
    # Set DEBUG to True to run on a small subset of data for quick testing
    DEBUG = False
    DEBUG_SAMPLES = 50  # Number of samples to use if DEBUG is True

    # ==========================================
    # Data Parameters
    # ==========================================
    # Resolution: 256 (H) x 640 (W)
    IMAGE_SIZE = (256, 640)

    # ImageNet Normalization Statistics
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    NUM_CLASSES = 19
    INPUT_CHANNELS = 3  # Channel replication (Gray -> RGB)

    # ==========================================
    # Model Parameters
    # ==========================================
    BACKBONE = "resnet34"
    PRETRAINED = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Stochastic Weight Averaging (SWA)
    # Teacher SWA: Active in final 25%
    TEACHER_SWA_START_EPOCH = int(NUM_EPOCHS * 0.75)
    # Student SWA: Active in final 30%
    STUDENT_SWA_START_EPOCH = int(NUM_EPOCHS * 0.70)
    SWA_LR = 1e-4

    # Distillation
    TEMPERATURE = 1.5

    # ==========================================
    # Stratified Policies (Augmentation)
    # ==========================================
    # Defines the specific regularization strategies for the ensemble
    # CoarseDropout params: max_holes, max_height, max_width
    STRATIFIED_POLICIES = {
        "Texture": {
            "description": "High Mixup, Low Cutout. Forces linearity and texture bias.",
            "mixup_alpha": 0.4,
            "cutout_prob": 0.2,
            "cutout_params": {"max_holes": 1, "max_height": 30, "max_width": 30},
        },
        "Feature": {
            "description": "Low Mixup, High Unstructured Cutout. Forces robust feature detection.",
            "mixup_alpha": 0.1,
            "cutout_prob": 0.8,
            "cutout_params": {"max_holes": 8, "max_height": 60, "max_width": 60},
        },
        "Balanced": {
            "description": "Standard Mixup and Cutout.",
            "mixup_alpha": 0.2,
            "cutout_prob": 0.5,
            "cutout_params": {"max_holes": 4, "max_height": 50, "max_width": 50},
        },
    }

    # The student model uses the Balanced policy
    STUDENT_POLICY_NAME = "Balanced"
