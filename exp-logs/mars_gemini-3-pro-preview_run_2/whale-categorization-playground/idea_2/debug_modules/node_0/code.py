import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Set random seeds for reproducibility
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -----------------------------------------------------------------------------
# 1. Import and Patch Library Modules for Speed
# -----------------------------------------------------------------------------
# We need to ensure the code runs fast (Debug mode, 1 Epoch, Small Batch)
# Since modules import variables using 'from config import var', we must patch
# the variables in the loaded modules directly.

import library.config
import library.dataset
import library.trainer
import library.inference
import library.model
import library.utils

print(">>> Patching configurations for fast demonstration...")

# Patch Global Config
library.config.DEBUG = True
library.config.NUM_EPOCHS = 1
library.config.BATCH_SIZE = 8
library.config.DEBUG_SIZE = 50  # Use very small subset for speed

# Patch Dataset Module
library.dataset.DEBUG = True
library.dataset.DEBUG_SIZE = 50
library.dataset.BATCH_SIZE = 8

# Patch Trainer Module
library.trainer.NUM_EPOCHS = 1
library.trainer.BATCH_SIZE = 8

# Patch Inference Module
library.inference.BATCH_SIZE = 8

print("Configuration patched: DEBUG=True, NUM_EPOCHS=1, BATCH_SIZE=8")

# -----------------------------------------------------------------------------
# 2. Verify Metric Calculation
# -----------------------------------------------------------------------------
print("\n>>> Verifying MAP@5 Metric Logic...")

# Case 1: Perfect prediction (Rank 1)
preds_1 = [["a", "b", "c", "d", "e"]]
targs_1 = ["a"]
score_1 = library.utils.calculate_map5(preds_1, targs_1)
assert np.isclose(score_1, 1.0), f"Expected 1.0, got {score_1}"

# Case 2: Prediction at Rank 2
preds_2 = [["a", "b", "c", "d", "e"]]
targs_2 = ["b"]
score_2 = library.utils.calculate_map5(preds_2, targs_2)
assert np.isclose(score_2, 0.5), f"Expected 0.5, got {score_2}"

# Case 3: Target not in top 5
preds_3 = [["a", "b", "c", "d", "e"]]
targs_3 = ["z"]
score_3 = library.utils.calculate_map5(preds_3, targs_3)
assert np.isclose(score_3, 0.0), f"Expected 0.0, got {score_3}"

print("MAP@5 Metric verification passed.")

# -----------------------------------------------------------------------------
# 3. Verify Data Loading and Pipeline
# -----------------------------------------------------------------------------
print("\n>>> Verifying Data Loading Pipeline...")

# Initialize DataLoaders
# This will trigger cache generation for the debug subset
train_loader, val_loader, test_loader, label_encoder = library.dataset.get_dataloaders(
    load_cached_data=False
)

# Check Train Loader
images, labels = next(iter(train_loader))
print(f"Train Batch Shape: {images.shape}")
print(f"Train Labels Shape: {labels.shape}")

assert images.shape[0] == library.config.BATCH_SIZE
assert images.shape[1] == 3  # RGB
assert images.shape[2] == library.config.IMAGE_SIZE
assert images.shape[3] == library.config.IMAGE_SIZE
assert isinstance(label_encoder, dict)
assert len(label_encoder) > 0

print("DataLoaders initialized and verified successfully.")

# -----------------------------------------------------------------------------
# 4. Verify Model Architecture
# -----------------------------------------------------------------------------
print("\n>>> Verifying Model Architecture...")

model = library.model.WhaleArcFaceModel(
    model_name="efficientnet_b0",
    num_classes=len(label_encoder),
    embedding_dim=512,
    pretrained=False,  # False for speed
)
model.to(library.config.DEVICE)
model.eval()

# Create dummy input
dummy_input = torch.randn(2, 3, 256, 256).to(library.config.DEVICE)
dummy_labels = torch.tensor([0, 1]).to(library.config.DEVICE)

# Test Inference Mode (No Labels) -> Should return Embeddings
with torch.no_grad():
    embeddings = model(dummy_input)
    print(f"Embeddings Shape: {embeddings.shape}")
    assert embeddings.shape == (2, 512)

# Test Training Mode (With Labels) -> Should return Logits
model.train()
logits = model(dummy_input, dummy_labels)
print(f"Logits Shape: {logits.shape}")
assert logits.shape == (2, len(label_encoder))

print("Model architecture verified successfully.")

# -----------------------------------------------------------------------------
# 5. Run Training Loop (Trainer)
# -----------------------------------------------------------------------------
print("\n>>> Running Training Loop (1 Epoch, Debug Subset)...")

# Initialize Trainer
# Note: Trainer re-initializes dataloaders internally, but uses our patched config
trainer = library.trainer.Trainer(load_cached_data=True)

# Run Fit
# This includes training, validation, and saving the best model
trainer.fit()

# Verify Output
best_model_path = os.path.join(library.config.WORKING_DIR, "best_model.pth")
assert os.path.exists(best_model_path), "Best model checkpoint was not created."
print(f"Training completed. Checkpoint saved at {best_model_path}")

# -----------------------------------------------------------------------------
# 6. Run Inference Pipeline
# -----------------------------------------------------------------------------
print("\n>>> Running Inference Pipeline...")

# Initialize Inference Manager
inference_manager = library.inference.InferenceManager(checkpoint_name="best_model.pth")

# Run Prediction
# This loads the model, computes embeddings for Gallery (Train) and Query (Test),
# performs KNN, and saves submission.csv
inference_manager.predict(load_cached_data=False, threshold=0.35)

# Verify Submission
submission_path = os.path.join(library.config.SUBMISSION_DIR, "submission.csv")
assert os.path.exists(submission_path), "Submission file was not created."

df_sub = pd.read_csv(submission_path)
print(f"Submission File Rows: {len(df_sub)}")
print("First 3 rows:")
print(df_sub.head(3))

# Verify Format
assert "Image" in df_sub.columns
assert "Id" in df_sub.columns
# Check if predictions are space-separated strings
sample_pred = df_sub.iloc[0]["Id"]
assert isinstance(sample_pred, str)
assert (
    len(sample_pred.split()) == 5
), f"Expected 5 predictions per image, got {len(sample_pred.split())}"

print("Inference pipeline verified successfully.")

print("\n=======================================================")
print("   ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
print("=======================================================")
