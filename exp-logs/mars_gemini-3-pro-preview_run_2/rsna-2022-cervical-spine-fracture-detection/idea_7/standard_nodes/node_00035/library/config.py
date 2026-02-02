import os
import torch


class Config:
    # =========================
    # Directories & Paths
    # =========================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"

    # Ensure working directory exists for caching and outputs
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Data Paths
    TRAIN_IMAGES_DIR = os.path.join(INPUT_ROOT, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_ROOT, "test_images")
    BOUNDING_BOX_PATH = os.path.join(INPUT_ROOT, "train_bounding_boxes.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_ROOT, "sample_submission.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # =========================
    # Data Preprocessing
    # =========================
    # 2.5D Stacking: Input is stack of (z-1, z, z+1)
    IN_CHANNELS = 3

    # Image Dimensions
    # Reduced to 256x256 to prevent OOM (Cite debug_lesson_1)
    IMAGE_SIZE = (256, 256)

    # Sequence Length
    # Reduced to 48 to manage effective batch size (Batch * Seq_Len)
    SEQ_LEN = 48

    # Dataloader
    NUM_WORKERS = 4

    # =========================
    # Model Architecture
    # =========================
    BACKBONE = "tf_efficientnet_b4_ns"

    # LSTM & Head Configuration
    HIDDEN_DIM = 256
    NUM_CLASSES = 8  # C1-C7 (7) + patient_overall (1)

    # Regularization
    DROP_RATE = 0.3
    DROP_PATH_RATE = 0.2

    # =========================
    # Training Hyperparameters
    # =========================
    SEED = 42

    # Batch Size
    # Reduced to 2 to prevent OOM
    BATCH_SIZE = 2

    EPOCHS = 10
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1000.0

    # Early Stopping
    PATIENCE = 3

    # =========================
    # Loss & Metric Configuration
    # =========================
    TARGET_COLUMNS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    # Weighted Multi-Label Log Loss Weights
    # "The any label is weighted more highly than specific fracture level sub-types"
    # Weight 7.0 for overall, 1.0 for each vertebrae
    LOSS_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0])

    # Positive Class Weight for Study Loss (to handle class imbalance)
    POS_WEIGHT_STUDY = 2.0

    # Weight for the Auxiliary Dense Slice Loss
    LAMBDA_SLICE_LOSS = 1.0

    # =========================
    # Compute
    # =========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
