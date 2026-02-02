import os
import torch

# -----------------------------------------------------------------------------
# Global Constants & Paths
# -----------------------------------------------------------------------------
SEED = 42

# Base directories
BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "input")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
WORKING_DIR = os.path.join(BASE_DIR, "working", "idea_39")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")

# Sub-directories in working
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
PREDICTIONS_DIR = os.path.join(WORKING_DIR, "predictions")

# Ensure writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

PATHS = {
    "input": INPUT_DIR,
    "metadata": METADATA_DIR,
    "working": WORKING_DIR,
    "submission": os.path.join(SUBMISSION_DIR, "submission.csv"),
    "checkpoints": CHECKPOINT_DIR,
    "cache": CACHE_DIR,
    "predictions": PREDICTIONS_DIR,
    "train_meta": os.path.join(METADATA_DIR, "train.csv"),
    "val_meta": os.path.join(METADATA_DIR, "val.csv"),
    "test_meta": os.path.join(METADATA_DIR, "test.csv"),
    "model_save_path": os.path.join(CHECKPOINT_DIR, "best_model.pth"),
}

# -----------------------------------------------------------------------------
# Gesture Vocabulary
# -----------------------------------------------------------------------------
GESTURE_MAP = {
    "vattene": 1,
    "vieniqui": 2,
    "perfetto": 3,
    "furbo": 4,
    "cheduepalle": 5,
    "chevuoi": 6,
    "daccordo": 7,
    "seipazzo": 8,
    "combinato": 9,
    "freganiente": 10,
    "ok": 11,
    "cosatifarei": 12,
    "basta": 13,
    "prendere": 14,
    "noncenepiu": 15,
    "fame": 16,
    "tantotempo": 17,
    "buonissimo": 18,
    "messidaccordo": 19,
    "sonostufo": 20,
}

ID_TO_GESTURE = {v: k for k, v in GESTURE_MAP.items()}

# -----------------------------------------------------------------------------
# Joint Configuration
# -----------------------------------------------------------------------------
# Full Kinect Skeleton Joints (20)
JOINT_NAMES = [
    "HipCenter",
    "Spine",
    "ShoulderCenter",
    "Head",
    "ShoulderLeft",
    "ElbowLeft",
    "WristLeft",
    "HandLeft",
    "ShoulderRight",
    "ElbowRight",
    "WristRight",
    "HandRight",
    "HipLeft",
    "KneeLeft",
    "AnkleLeft",
    "FootLeft",
    "HipRight",
    "KneeRight",
    "AnkleRight",
    "FootRight",
]

JOINT_INDICES = {name: i for i, name in enumerate(JOINT_NAMES)}

# Selected 12 Upper-Body Joints
SELECTED_JOINTS = [
    "HipCenter",
    "Spine",
    "ShoulderCenter",
    "Head",
    "ShoulderLeft",
    "ElbowLeft",
    "WristLeft",
    "HandLeft",
    "ShoulderRight",
    "ElbowRight",
    "WristRight",
    "HandRight",
]

SELECTED_JOINT_INDICES = [JOINT_INDICES[j] for j in SELECTED_JOINTS]


# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
def get_hyperparams(debug=False):
    """
    Returns the hyperparameters dictionary.
    Args:
        debug (bool): If True, adjusts parameters for a quick debugging run.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Base configuration
    hp = {
        # General
        "seed": SEED,
        "device": device,
        "num_workers": 4,
        "debug": debug,
        # Data Preprocessing
        "scale_factor": 0.001,  # Convert mm to meters
        "selected_indices": SELECTED_JOINT_INDICES,
        "audio_mfcc_dim": 13,
        # Model Architecture (MG-CRGN)
        "stem_kernels": [3, 7, 11],
        "lstm_hidden_size": 256,
        "lstm_layers": 2,
        "lstm_bidirectional": True,
        "tcn_channels": 256,
        "tcn_layers": 10,
        "tcn_kernel_size": 3,
        "tcn_dropout": 0.3,
        "num_classes": 21,  # 0=Background, 1-20=Gestures
        # Training
        "epochs": 50,
        "batch_size": 8,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "grad_clip": 5.0,
        "patience": 10,  # Early stopping patience
        # Loss Weights
        # Class 0 (Background) gets 0.1, others get 1.0
        "class_weights": [0.1] + [1.0] * 20,
        "lambda_cls": 1.0,
        "lambda_bnd": 1.0,
        "lambda_smooth": 0.15,
        # Augmentation
        "aug_sigma": 0.01,
        "aug_temp_filter_width": 5,
    }

    # Adjust for debugging
    if debug:
        hp["epochs"] = 2
        hp["batch_size"] = 2
        hp["num_workers"] = 0
        hp["sample_size"] = 10  # Only use 10 samples for debugging
    else:
        hp["sample_size"] = None  # Use full dataset

    return hp
