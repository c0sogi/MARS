import os
import torch


class Config:
    # ==========================================
    # General Configuration
    # ==========================================
    PROJECT_NAME = "apple_disease_detection"
    IDEA_NAME = "idea_14"
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compute Environment
    # 12 vCPUs available. Setting workers to 4-6 is usually optimal to avoid overhead.
    NUM_WORKERS = 4

    # ==========================================
    # Directories & Paths
    # ==========================================
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = f"./working/{IDEA_NAME}"

    # Output subdirectories
    OUTPUT_DIR = os.path.join(WORKING_DIR, "output")
    SUBMISSION_DIR = "./submission"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Data File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    IMAGES_DIR = os.path.join(INPUT_DIR, "images")
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Parameters
    # ==========================================
    IMG_SIZE = 256
    # Batch size 64 fits comfortably on A100 with ResNet34 @ 256x256
    BATCH_SIZE = 64
    NUM_CLASSES = 4
    CLASS_LABELS = ["healthy", "multiple_diseases", "rust", "scab"]

    # ImageNet Normalization Statistics
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "resnet34"
    PRETRAINED = True
    # Simple head: Global Average Pooling + Linear. No high dropout needed for ResNet34 fine-tuning.
    DROPOUT_RATE = 0.0

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 12  # Budget for Phase 1 to find optimal epoch
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 10.0

    # Scheduler: Cosine Annealing Warm Restarts
    # T_0 synchronized with total epochs to ensure full convergence curve
    T_0 = 12  # Cite {solution_lesson_node_00015}
    T_MULT = 1
    ETA_MIN = 1e-6

    # ==========================================
    # Strategy: Calibrated Full-Data Seed Ensemble
    # ==========================================
    # Phase 1: Proxy Calibration
    N_FOLDS = 5

    # Phase 2: Production Training (Seed Ensemble)
    # Using 5 distinct seeds for the ensemble members
    PHASE2_SEEDS = [42, 101, 2022, 999, 12345]

    # Initialization Verification
    # Threshold: -ln(1/4) approx 1.386.
    # If initial loss > 1.38, the model head is not initialized correctly relative to the backbone.
    INIT_LOSS_THRESHOLD = 1.38

    # Loss Function
    USE_CLASS_WEIGHTS = (
        True  # To handle class imbalance (healthy/rust/scab vs multiple_diseases)
    )

    # ==========================================
    # Augmentation
    # ==========================================
    AUG_PROB = 0.5
    # Specific limits for Albumentations
    BRIGHTNESS_LIMIT = 0.2
    CONTRAST_LIMIT = 0.2
    ROTATE_LIMIT = (
        30  # Increased to match domain invariances Cite {solution_lesson_node_00016}
    )
    SCALE_LIMIT = 0.1

    # Test Time Augmentation (TTA)
    # Candidates: HorizontalFlip, VerticalFlip
    # Logic: Applied only if Phase 1 OOF metrics improve
    TTA_TRANSFORMS = ["horizontal_flip", "vertical_flip"]
