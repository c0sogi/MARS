import os
import torch


class Config:
    """
    Configuration for Idea 29: Corrected Augmentation-Stratified ResNet-34 Ensemble Distillation.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", "idea_29")

    # Ensure the working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Data Source
    SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")

    # Output Paths
    TEACHER_1_CHECKPOINT = os.path.join(WORKING_DIR, "teacher_1_swa.pth")
    TEACHER_2_CHECKPOINT = os.path.join(WORKING_DIR, "teacher_2_swa.pth")
    TEACHER_3_CHECKPOINT = os.path.join(WORKING_DIR, "teacher_3_swa.pth")
    STUDENT_CHECKPOINT = os.path.join(WORKING_DIR, "student_swa.pth")

    PSEUDO_LABEL_PATH = os.path.join(WORKING_DIR, "pseudo_labels.parquet")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Global Settings
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging / Development
    DEBUG = False
    MAX_SAMPLES = None  # Set to int (e.g., 100) to limit dataset size for quick testing

    # =========================================================================
    # Data Preprocessing
    # =========================================================================
    # High-Fidelity Resolution: Preserves frequency resolution and temporal morphology
    IMG_HEIGHT = 256
    IMG_WIDTH = 640
    IN_CHANNELS = 3  # Channel replication (Grayscale -> RGB)

    # ImageNet Normalization Statistics
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_BACKBONE = "resnet34"
    PRETRAINED = True
    NUM_CLASSES = 19

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    NUM_EPOCHS = 50

    # Optimizer: AdamW (Correcting previous failure with SGD)
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-2

    # Stochastic Weight Averaging (SWA)
    SWA_LR = 1e-4

    # Teacher SWA: Active in final 25% (Last ~12 epochs) -> Start at Epoch 38
    TEACHER_SWA_START_EPOCH = 38

    # Student SWA: Active in final 30% (Last ~15 epochs) -> Start at Epoch 35
    STUDENT_SWA_START_EPOCH = 35

    # =========================================================================
    # Distillation Parameters
    # =========================================================================
    TEMPERATURE = 1.5  # Softens probability distribution

    # =========================================================================
    # Augmentation Policies (Stratification Strategy)
    # =========================================================================

    # Teacher 1: Linearity Bias
    # Strategy: High Mixup to enforce linear behavior between classes, Low Cutout.
    POLICY_TEACHER_1 = {
        "mixup_alpha": 0.4,
        "cutout_params": {"num_holes": 1, "max_h_size": 32, "max_w_size": 32, "p": 0.5},
    }

    # Teacher 2: Occlusion Robustness
    # Strategy: Low Mixup, High Unstructured Cutout (many small holes) to force feature redundancy.
    POLICY_TEACHER_2 = {
        "mixup_alpha": 0.1,
        "cutout_params": {
            "num_holes": 20,
            "max_h_size": 16,
            "max_w_size": 16,
            "p": 0.8,
        },
    }

    # Teacher 3 & Student: Balanced
    # Strategy: Standard regularization settings.
    POLICY_BALANCED = {
        "mixup_alpha": 0.2,
        "cutout_params": {"num_holes": 5, "max_h_size": 24, "max_w_size": 24, "p": 0.5},
    }
