import os
import torch

# ==========================================
# Directories and Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_2"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Sample Submission
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# ==========================================
# System Configuration
# ==========================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# We have 12 vCPUs, so 8 workers is a safe number for DataLoader
NUM_WORKERS = 8

# ==========================================
# Model Configuration
# ==========================================
# Backbone A: ConvNeXt Large (CNN)
# Using a strong ImageNet-21k pretrained model fine-tuned on 1k
MODEL_CNN = "convnext_large.fb_in22k_ft_in1k"

# Backbone B: ViT Large (Transformer)
# Using a strong ImageNet-21k pretrained model fine-tuned on 1k
MODEL_VIT = "vit_large_patch16_224.augreg_in21k_ft_in1k"

# Input Image Size (Standard for these models)
IMAGE_SIZE = 224

# Number of classes in the dataset
NUM_CLASSES = 120

# ==========================================
# Training / Inference Hyperparameters
# ==========================================
# Batch size for feature extraction (A100 40GB can handle large batches)
BATCH_SIZE = 128

# Logistic Regression Hyperparameters
# C: Inverse of regularization strength (smaller = stronger regularization)
# max_iter: Maximum number of iterations for the solver
LOGREG_C = 1.0
LOGREG_MAX_ITER = 1000
LOGREG_SOLVER = "lbfgs"

# ==========================================
# Caching Paths
# ==========================================
# Define paths for cached embeddings to ensure consistency across modules
CNN_TRAIN_EMBEDDINGS = os.path.join(WORKING_DIR, "cnn_train_embeddings.npy")
CNN_VAL_EMBEDDINGS = os.path.join(WORKING_DIR, "cnn_val_embeddings.npy")
CNN_TEST_EMBEDDINGS = os.path.join(WORKING_DIR, "cnn_test_embeddings.npy")

VIT_TRAIN_EMBEDDINGS = os.path.join(WORKING_DIR, "vit_train_embeddings.npy")
VIT_VAL_EMBEDDINGS = os.path.join(WORKING_DIR, "vit_val_embeddings.npy")
VIT_TEST_EMBEDDINGS = os.path.join(WORKING_DIR, "vit_test_embeddings.npy")

LABELS_TRAIN_PATH = os.path.join(WORKING_DIR, "train_labels.npy")
LABELS_VAL_PATH = os.path.join(WORKING_DIR, "val_labels.npy")
IDS_TEST_PATH = os.path.join(WORKING_DIR, "test_ids.npy")
