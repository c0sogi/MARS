import os
from pathlib import Path

# --- Directory Paths ---
INPUT_DIR = Path("./input")
METADATA_DIR = Path("./metadata")
WORKING_DIR = Path("./working/idea_2")
SUBMISSION_DIR = Path("./submission")

# Ensure necessary directories exist
WORKING_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# --- File Paths ---
# Using the parquet metadata files as the primary data source
TRAIN_DATA_PATH = METADATA_DIR / "train.parquet"
VAL_DATA_PATH = METADATA_DIR / "val.parquet"
TEST_DATA_PATH = METADATA_DIR / "test.parquet"
SUBMISSION_PATH = SUBMISSION_DIR / "submission.csv"

# --- Column Definitions ---
USER_COL = "customer_id"
ITEM_COL = "article_id"
DATE_COL = "t_dat"
PRICE_COL = "price"
IMAGE_PATH_COL = "image_path"

# --- Hyperparameters ---
RANDOM_STATE = 42
HISTORY_WEEKS = (
    12  # Increased to 12 weeks to capture more trend data while maintaining relevance
)
TOP_K = 12  # Number of items to predict per customer (MAP@12 metric)
DECAY_RATE = 2.5  # Power for time decay weighting (heuristic for fashion trends)

# --- Compute Configuration ---
NUM_WORKERS = 12  # Available vCPUs
