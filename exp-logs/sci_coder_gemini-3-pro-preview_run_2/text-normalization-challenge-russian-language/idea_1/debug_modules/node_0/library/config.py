import os

# ==========================================
# Directory Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_1"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# File Path Configuration
# ==========================================
# Input Data (Generated Metadata)
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Sample Submission (Raw Input)
# Using the file listed in the dataset information
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "ru_sample_submission_2.csv")

# Output Submission
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Files (Processed Data & Model)
# Using parquet for dataframes and npy for model dictionaries
TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")
MODEL_CACHE_PATH = os.path.join(WORKING_DIR, "hfbb_model.npy")

# ==========================================
# Column Definitions
# ==========================================
COL_SENTENCE_ID = "sentence_id"
COL_TOKEN_ID = "token_id"
COL_BEFORE = "before"
COL_AFTER = "after"
COL_CLASS = "class"
COL_ID = "id"

# ==========================================
# Model Constants & Special Tokens
# ==========================================
TOKEN_BOS = "<BOS>"
TOKEN_EOS = "<EOS>"

# Global Random Seed
SEED = 42
