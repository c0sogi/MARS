import os

# =============================================================================
# DIRECTORIES AND PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_13"

# Ensure working directory exists for caching and checkpoints
os.makedirs(WORKING_DIR, exist_ok=True)

TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = "./submission/submission.csv"

# =============================================================================
# AUDIO PARAMETERS
# =============================================================================
# Standard Google Speech Commands parameters
SAMPLE_RATE = 16000
DURATION = 1.0  # seconds
AUDIO_LEN = int(SAMPLE_RATE * DURATION)

# Spectrogram generation (High-Fidelity)
N_MELS = 128
N_FFT = 1024  # ~64ms window
HOP_LENGTH = 160  # ~10ms step
F_MIN = 20
F_MAX = SAMPLE_RATE // 2

# =============================================================================
# LABEL DEFINITIONS
# =============================================================================
# The 12 Target Labels for the competition metric
TARGET_LABELS = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"}
SILENCE_LABEL = "silence"
UNKNOWN_LABEL = "unknown"

# Fine-Grained Source Labels (36 Classes)
# Includes Targets, Silence, and Auxiliary words from GSC v2 dataset.
# The model predicts these to maintain decision boundary precision.
FINE_GRAINED_LABELS = sorted(
    [
        # Targets
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
        # Silence
        "silence",
        # Auxiliary / Unknown words
        "backward",
        "bed",
        "bird",
        "cat",
        "dog",
        "eight",
        "five",
        "follow",
        "forward",
        "four",
        "happy",
        "house",
        "learn",
        "marvin",
        "nine",
        "one",
        "seven",
        "sheila",
        "six",
        "three",
        "tree",
        "two",
        "visual",
        "wow",
        "zero",
    ]
)

# Mappings for Model Head
LABEL2ID = {label: i for i, label in enumerate(FINE_GRAINED_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(FINE_GRAINED_LABELS)}
NUM_CLASSES = len(FINE_GRAINED_LABELS)

# =============================================================================
# MODEL & TRAINING HYPERPARAMETERS
# =============================================================================
MODEL_NAME = "efficientnet_b2"
IN_CHANNELS = 1  # Spectrogram input

SEED = 42
BATCH_SIZE = 128  # Optimized for A100
NUM_WORKERS = 12  # Optimized for 12 vCPUs
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10  # Early stopping patience

# Augmentation
MIXUP_ALPHA = 1.0
NOISE_SNR_MIN = 10  # dB
NOISE_SNR_MAX = 30  # dB

# Self-Training
CONFIDENCE_THRESHOLD = 0.95


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_source_label(filepath):
    """
    Extracts the fine-grained label from the filepath.
    Handles the special '_background_noise_' folder mapping to 'silence'.

    Args:
        filepath (str): Relative path, e.g., 'train/audio/bed/001.wav'

    Returns:
        str: The label name (e.g., 'bed', 'silence').
    """
    parts = filepath.split(os.sep)
    # Typical path: train/audio/<label>/<file>
    # Parent folder is at index -2
    if len(parts) < 2:
        return UNKNOWN_LABEL

    folder_name = parts[-2]

    if folder_name == "_background_noise_":
        return SILENCE_LABEL

    # If the folder is in our known list, return it.
    # Otherwise, it might be a new unknown word (though unlikely in this dataset).
    if folder_name in LABEL2ID:
        return folder_name

    # Fallback for unexpected folders (mapped to unknown in training logic usually)
    return folder_name


def map_prediction_to_submission(predicted_label):
    """
    Maps the fine-grained prediction to the 12-class competition format.

    Args:
        predicted_label (str): The label predicted by the model (e.g., 'bed', 'yes').

    Returns:
        str: One of {'yes', 'no', ..., 'stop', 'go', 'silence', 'unknown'}
    """
    if predicted_label in TARGET_LABELS:
        return predicted_label
    if predicted_label == SILENCE_LABEL:
        return SILENCE_LABEL
    return UNKNOWN_LABEL
