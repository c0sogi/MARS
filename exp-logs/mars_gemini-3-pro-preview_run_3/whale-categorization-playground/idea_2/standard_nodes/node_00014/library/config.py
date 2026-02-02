import os
import torch

# -----------------------------------------------------------------------------
# Global Configuration for Whale Species Identification
# -----------------------------------------------------------------------------

# Reproducibility
SEED = 42

# -----------------------------------------------------------------------------
# Compute Environment
# -----------------------------------------------------------------------------
# Use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Number of data loading workers (utilizing available vCPUs)
NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# File Paths & Directories
# -----------------------------------------------------------------------------
# Input Base Directory (Read-Only)
INPUT_DIR = "./input"

# Metadata Directory (Pre-generated CSVs)
METADATA_DIR = "./metadata"
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Working Directory (Write Allowed)
# Stores checkpoints, cache files, and intermediate results
WORKING_DIR = "./working/idea_3"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission Directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Output File Paths
MODEL_PATH = os.path.join(WORKING_DIR, "efficientnet_arcface.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Paths (for embeddings and labels to speed up inference/analysis)
TRAIN_EMBEDDINGS_CACHE = os.path.join(WORKING_DIR, "train_embeddings.npy")
TRAIN_LABELS_CACHE = os.path.join(WORKING_DIR, "train_labels.npy")
TEST_EMBEDDINGS_CACHE = os.path.join(WORKING_DIR, "test_embeddings.npy")
TEST_NAMES_CACHE = os.path.join(WORKING_DIR, "test_names.npy")

# -----------------------------------------------------------------------------
# Data Hyperparameters
# -----------------------------------------------------------------------------
# Input image size (Height, Width)
# Reduced from 448 to 384 to prevent CUDA OOM on 16GB GPU
IMAGE_SIZE = (384, 384)

# Batch Size
# Reduced from 32 to 16 to prevent CUDA OOM on 16GB GPU
BATCH_SIZE = 16

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
# Number of training epochs
NUM_EPOCHS = 30

# Learning Rate (AdamW)
LEARNING_RATE = 3e-4

# Weight Decay for regularization
WEIGHT_DECAY = 1e-4

# -----------------------------------------------------------------------------
# ArcFace Model Hyperparameters
# -----------------------------------------------------------------------------
# Dimension of the embedding vector
EMBEDDING_SIZE = 512

# Angular Margin (m)
# The penalty added to the angle of the ground truth class.
# 0.5 is the standard recommended value for ArcFace.
ARC_MARGIN = 0.50

# Feature Scale (s)
# The radius of the hypersphere.
# 30.0 is a typical value for this parameter.
ARC_SCALE = 30.0

# -----------------------------------------------------------------------------
# Inference Hyperparameters
# -----------------------------------------------------------------------------
# Confidence Threshold for Open-Set Recognition
# If the cosine similarity between a test image and its nearest training neighbor
# is below this threshold, the image is classified as 'new_whale'.
# Cosine similarity range: [-1.0, 1.0]
CONFIDENCE_THRESHOLD = 0.40
