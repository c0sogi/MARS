import os
import torch


class Config:
    """
    Configuration for the Breast Cancer Detection Task.
    Implements settings for 'Idea 3': Analytically Calibrated EfficientNetV2 with ROI-Aware Preprocessing.
    """

    # ==========================================================================
    # 1. Paths & Directories
    # ==========================================================================
    # Base Input Directory (Read-Only)
    INPUT_DIR = "./input"

    # Metadata Directory (Generated in previous steps)
    METADATA_DIR = "./metadata"

    # Working Directory (For intermediate files, checkpoints, cache)
    # We use a specific subdirectory for this idea
    WORKING_DIR = "./working/idea_3"

    # Submission Directory
    SUBMISSION_DIR = "./submission"

    # Ensure writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Final Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================================================
    # 2. Data Preprocessing & Input
    # ==========================================================================
    # Input Image Dimensions
    # 640x640 provides a good balance between resolution and compute for EfficientNetV2-S
    IMG_HEIGHT = 640
    IMG_WIDTH = 640
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

    # ROI (Region of Interest) Processing
    # Parameters to crop the breast tissue from the background
    ROI_IGNORE_BLACK = True
    ROI_BINARIZE_THRESHOLD = 0.05  # Threshold to separate tissue from background

    # Normalization (ImageNet Defaults)
    # Used after converting single-channel DICOM to 3-channel RGB
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ==========================================================================
    # 3. Model Architecture
    # ==========================================================================
    # Backbone: EfficientNetV2-Small
    # Selected for high efficiency and strong performance on texture-based tasks
    MODEL_NAME = "efficientnet_v2_s"
    PRETRAINED = True

    # Model Head
    NUM_CLASSES = 1  # Binary classification (Cancer / No Cancer)
    IN_CHANNELS = 3  # Model expects RGB
    DROPOUT_RATE = 0.2  # Classifier dropout
    DROP_PATH_RATE = 0.1  # Stochastic depth rate

    # ==========================================================================
    # 4. Training Hyperparameters
    # ==========================================================================
    # Reproducibility
    SEED = 42

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 8  # Number of data loading workers (12 vCPUs available)

    # Optimization
    BATCH_SIZE = 32  # Fits in A100 40GB with 640x640 resolution
    EPOCHS = 10  # Number of training epochs
    LEARNING_RATE = 1e-4  # Initial learning rate
    WEIGHT_DECAY = 1e-2  # Weight decay for AdamW

    # Learning Rate Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 3  # Epochs to wait before stopping if val_loss doesn't improve

    # Balanced Sampling Strategy
    # We force the training batches to be balanced to ensure gradient flow for the minority class.
    # This distorts the model's probabilistic output, which we correct analytically later.
    POS_RATIO = 0.5  # Target ratio of positive samples in each batch

    # ==========================================================================
    # 5. Analytical Prior Correction
    # ==========================================================================
    # These constants are used to adjust the logits at inference time.
    # Formula: logit_corrected = logit_pred - log(P_train/(1-P_train)) + log(P_test/(1-P_test))

    # P_TRAIN: The prevalence the model "sees" during training (0.5 due to balanced sampling)
    P_TRAIN = 0.5

    # P_TEST: The natural prevalence expected in the test set
    # Based on training data analysis (~2.07%) and sample submission (~2.06%)
    P_TEST = 0.0206

    # ==========================================================================
    # 6. Debugging & Development
    # ==========================================================================
    # Set DEBUG to True to run on a small subset of data for rapid iteration
    DEBUG = False
    DEBUG_SUBSET_SIZE = 1000  # Number of samples to use in debug mode
