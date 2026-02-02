import os
import torch


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_11"
    SUBMISSION_PATH = "./submission.csv"

    # Ensure working directory exists immediately upon import
    os.makedirs(WORKING_DIR, exist_ok=True)

    # =========================================================================
    # Data Generation & Preprocessing
    # =========================================================================
    TILE_SIZE = 512
    STRIDE = 512  # Non-overlapping tiling for training

    # Z-Axis / Slab Configuration (Overlapping Thick Slab)
    # Strategy: Fixed Narrow Context (slices 20-44)
    # We construct 3 input channels using Maximum Intensity Projection (MIP).
    # Channel 1: Slices 20-32 (MIP)
    # Channel 2: Slices 26-38 (MIP)
    # Channel 3: Slices 32-44 (MIP)
    Z_START = 20  # The anchor start slice for the training slab
    Z_DIM = 12  # Depth of each MIP channel
    Z_STEP = 6  # Overlap stride between channels

    # Normalization Constants
    PIXEL_MIN = 0.0
    PIXEL_MAX = 65535.0  # 16-bit TIFF data

    # =========================================================================
    # Inference Strategy: Decoupled Z-Scanning
    # =========================================================================
    # To capture "wandering" ink strokes without training on noisy data,
    # we scan multiple depths at inference time and max-fuse the predictions.
    # Offsets are relative to Z_START (20).
    # Scan A: Start 18 (-2) -> Covers 18-42
    # Scan B: Start 20 (0)  -> Covers 20-44 (Same as training)
    # Scan C: Start 22 (+2) -> Covers 22-46
    INFERENCE_Z_OFFSETS = [-2, 0, 2]

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_ENCODER = "mit_b3"  # SegFormer B3
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 3
    NUM_CLASSES = 1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42

    # Compute Resources
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Batching & Optimization
    BATCH_SIZE = 16  # Optimized for A100 40GB with 512x512 inputs
    NUM_EPOCHS = 15
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6

    # Loss Function Configuration
    # We use a combination of BCE and Dice Loss

    # =========================================================================
    # Validation & Submission Logic
    # =========================================================================
    # Metric: F0.5 Score (Weights precision higher than recall)
    F_BETA = 0.5
    MASK_THRESHOLD = 0.5

    # Validation Gating
    # The submission file is only generated if the best validation F0.5 score
    # exceeds this baseline.
    BASELINE_SCORE = 0.5976

    # =========================================================================
    # Debugging / Development
    # =========================================================================
    # Set DEBUG to True to run on a small subset of data for quick pipeline checks
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200
