import os

# =============================================================================
# Directories
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Working directory specific to Idea 12 to avoid conflicts with other runs
WORKING_DIR = os.path.join(".", "working", "idea_12")
SUBMISSION_DIR = "./submission"

# =============================================================================
# File Paths
# =============================================================================
# Metadata files (Stratified splits with file paths)
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Raw Input Files
RAW_TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
RAW_TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Submission
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# Global Constants
# =============================================================================
SEED = 42
N_CLASSES = 99
N_FEATURES = 192

# Column Names
ID_COL = "id"
TARGET_COL = "species"
FILE_PATH_COL = "file_path"

# Numerical Stability for Log Loss
EPSILON = 1e-15


# =============================================================================
# Schema & Feature Definitions
# =============================================================================
def get_ordered_feature_list(include_types=None):
    """
    Returns the strict alphanumeric sequence of feature names.
    This enforces a deterministic schema for model training and inference,
    preventing implicit column reordering by pandas or other libraries.

    Args:
        include_types (list, optional): List of feature types to include.
                                        Defaults to ['margin', 'shape', 'texture'].

    Returns:
        list[str]: A sorted list of feature column names.
                   Example order: ['margin_1', 'margin_10', ..., 'margin_2', ...]
    """
    if include_types is None:
        include_types = ["margin", "shape", "texture"]

    features = []

    # Generate all feature names: type1 to type64
    for f_type in include_types:
        for i in range(1, 65):
            features.append(f"{f_type}{i}")

    # Apply strict alphanumeric sort.
    # This is critical: 'margin10' must consistently come before 'margin2'
    # if that is the sorted order, to match the vector positions.
    return sorted(features)


def get_directories():
    """
    Returns a list of directories that should exist for the project.
    """
    return [WORKING_DIR, SUBMISSION_DIR, METADATA_DIR]
