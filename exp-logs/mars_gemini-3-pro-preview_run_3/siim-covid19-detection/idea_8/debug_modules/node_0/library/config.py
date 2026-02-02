import os
import torch


class Config:
    """
    Configuration for Idea 8: Multi-Task Deformable DETR with Unified Query Interaction.
    """

    # =========================================================================
    # General Setup
    # =========================================================================
    PROJECT_NAME = "idea_8_multitask_deformable_detr"
    SEED = 42

    # Debugging / Development flags
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_DATA_SIZE = 100  # Number of samples to use when DEBUG is True

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Sample Submission
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpoint Path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Image Resizing Strategy: Letterbox (preserve aspect ratio)
    IMG_SIZE = 800  # Longest dimension target

    # DataLoader settings
    BATCH_SIZE = 8  # A100 40GB can handle this with Deformable DETR + ResNet50
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # Label Definitions
    # Study Level Labels
    STUDY_LABELS = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]
    NUM_STUDY_CLASSES = len(STUDY_LABELS)
    STUDY_LABEL2ID = {label: i for i, label in enumerate(STUDY_LABELS)}
    ID2STUDY_LABEL = {i: label for i, label in enumerate(STUDY_LABELS)}

    # Image Level Labels
    # For DETR, we typically have N object classes.
    # Here we have 1 foreground class: "opacity".
    # Background is handled by the model architecture (e.g., class index 0 or N).
    NUM_OBJECT_CLASSES = 1

    # =========================================================================
    # Model Architecture (Multi-Task Deformable DETR)
    # =========================================================================
    BACKBONE = "resnet50"

    # Transformer Parameters
    HIDDEN_DIM = 256
    NHEADS = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1

    # Deformable Attention Specifics
    NUM_FEATURE_LEVELS = 4
    ENC_N_POINTS = 4
    DEC_N_POINTS = 4

    # Query Configuration
    NUM_OBJECT_QUERIES = 100  # Standard object detection queries
    NUM_STUDY_QUERIES = 1  # Unified query for study-level classification

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Training Duration
    EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 5

    # Optimization
    # Transformers typically require different LRs for backbone and head
    LR_BACKBONE = 1e-5
    LR_TRANSFORMER = 1e-4
    WEIGHT_DECAY = 1e-4
    CLIP_MAX_NORM = 0.1  # Gradient clipping

    # Loss Weights (Hungarian Matching & Final Loss)
    # Costs for Bipartite Matching
    COST_CLASS = 2.0
    COST_BBOX = 5.0
    COST_GIOU = 2.0

    # Weights for the final loss components
    LOSS_CE_STUDY = 1.0  # Weight for the study classification task
    LOSS_CLASS = 2.0  # Focal loss weight for object classification
    LOSS_BBOX = 5.0  # L1 loss weight for box coordinates
    LOSS_GIOU = 2.0  # GIoU loss weight

    # =========================================================================
    # Inference & Post-Processing
    # =========================================================================
    # Confidence threshold for including an opacity in the prediction string
    POST_PROCESS_CONF_THRESH = 0.2

    # If study prediction is 'Negative for Pneumonia', force image prediction to 'none'
    FORCE_NONE_ON_NEGATIVE = True
