import os
import torch


class Config:
    # ==== Reproducibility ====
    SEED = 42

    # ==== Paths ====
    # Input Data (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory (Write Allowed)
    # Using idea_3 to fix coordinate system mismatch
    WORKING_DIR = "./working/idea_3"
    CACHE_DIR = WORKING_DIR
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "dla34_best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # ==== Data Processing / BEV Generation ====
    # Voxel / Grid settings
    # Range: (min, max) in meters
    X_RANGE = (-100, 100)
    Y_RANGE = (-100, 100)
    Z_RANGE = (-5, 3)

    # Resolution in meters per pixel
    BEV_RESOLUTION = 0.8

    # Resulting Grid Size: (200 / 0.8) = 250
    # Input size (W, H)
    # Adjusted to 256 to be divisible by 32 (network stride)
    INPUT_SIZE = (256, 256)

    # Input Channels for the model (Density, Intensity, Height)
    IN_CHANNELS = 3

    # ==== Model Architecture ====
    BACKBONE = "dla34"

    # Number of classes based on EDA (car, truck, bus, etc.)
    # Classes: car, other_vehicle, pedestrian, bicycle, truck, bus, motorcycle, animal, emergency_vehicle
    NUM_CLASSES = 9

    # Head configuration
    # Key: Name of head, Value: Number of output channels
    # heatmap: num_classes (classification)
    # reg: 2 (offset x, offset y)
    # wh: 3 (width, length, height) - log encoded or raw
    # depth: 1 (z coordinate)
    # rot: 2 (sin(yaw), cos(yaw))
    HEADS = {"hm": NUM_CLASSES, "reg": 2, "wh": 3, "depth": 1, "rot": 2}

    # Channel width for head intermediate layers
    HEAD_CONV = 256

    # ==== Training Strategy ====
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on vCPUs (12 available)

    # Optimization
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4

    # Scheduling
    NUM_EPOCHS = 25  # Minimum 25 as per plan
    PATIENCE = 5  # Early stopping patience

    # Loss Weights
    # Prioritizing geometric exactness by keeping regression weights high
    LOSS_WEIGHTS = {"hm": 1.0, "reg": 1.0, "wh": 1.0, "depth": 1.0, "rot": 1.0}

    # ==== Inference / Post-processing ====
    # Max number of detections per sample
    MAX_DETECTIONS = 50
    # Score threshold for filtering predictions
    SCORE_THRESHOLD = 0.2
    # IoU thresholds for evaluation (metric definition)
    IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # Class mapping (Index to Name)
    # Based on EDA frequency order or fixed mapping.
    # Note: Training usually maps class names to indices 0-8.
    CLASS_NAMES = [
        "car",
        "other_vehicle",
        "pedestrian",
        "bicycle",
        "truck",
        "bus",
        "motorcycle",
        "animal",
        "emergency_vehicle",
    ]
    CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
    ID_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

    @staticmethod
    def print_config():
        print("==== Configuration ====")
        print(f"Device: {Config.DEVICE}")
        print(f"Input Size: {Config.INPUT_SIZE}")
        print(f"Resolution: {Config.BEV_RESOLUTION} m/px")
        print(f"Backbone: {Config.BACKBONE}")
        print(f"Epochs: {Config.NUM_EPOCHS}")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Learning Rate: {Config.LEARNING_RATE}")
        print(f"Patience: {Config.PATIENCE}")
        print("=======================")
