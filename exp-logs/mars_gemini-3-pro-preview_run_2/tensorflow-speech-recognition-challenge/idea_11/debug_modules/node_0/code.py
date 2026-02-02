import os
import sys
import pandas as pd
import torch
import numpy as np
import shutil

# =============================================================================
# 1. SETUP AND CONFIGURATION OVERRIDE
# =============================================================================
# We import library.config first to modify settings before other modules load them.
import library.config

# Define a temporary working directory for this demo
DEMO_WORKING_DIR = "./working/demo_execution"
if os.path.exists(DEMO_WORKING_DIR):
    shutil.rmtree(DEMO_WORKING_DIR)
os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

# Override Config Paths and Hyperparameters for Speed
print("[Demo] Overriding configuration for rapid execution...")
library.config.WORKING_DIR = DEMO_WORKING_DIR
library.config.EPOCHS = 1
library.config.BATCH_SIZE = 4
library.config.NUM_WORKERS = (
    0  # Use 0 workers to avoid multiprocessing overhead in demo
)
library.config.DEBUG = False  # We handle subsetting manually via metadata files

# Create subset metadata files to limit dataset size
# We read the actual metadata to get valid file paths
original_train = pd.read_csv("./metadata/train.csv")
original_val = pd.read_csv("./metadata/val.csv")
original_test = pd.read_csv("./metadata/test.csv")

# Take small samples (ensure we have enough for a batch)
subset_train = original_train.head(32)
subset_val = original_val.head(16)
subset_test = original_test.head(16)

# Save temporary metadata
temp_train_path = os.path.join(DEMO_WORKING_DIR, "temp_train.csv")
temp_val_path = os.path.join(DEMO_WORKING_DIR, "temp_val.csv")
temp_test_path = os.path.join(DEMO_WORKING_DIR, "temp_test.csv")

subset_train.to_csv(temp_train_path, index=False)
subset_val.to_csv(temp_val_path, index=False)
subset_test.to_csv(temp_test_path, index=False)

# Update config to point to these new files
library.config.TRAIN_METADATA_PATH = temp_train_path
library.config.VAL_METADATA_PATH = temp_val_path
library.config.TEST_METADATA_PATH = temp_test_path

print(
    f"[Demo] Created subset metadata: Train={len(subset_train)}, Val={len(subset_val)}, Test={len(subset_test)}"
)

# =============================================================================
# 2. IMPORT LIBRARY MODULES
# =============================================================================
# Now we import the rest of the library. They will pick up the modified config values.
from library.dataset import SpeechDataset, get_dataloaders
from library.model import AudioEfficientNet
from library.trainer import Trainer
from library.utils import set_seed, save_submission

# Set seed for reproducibility
set_seed(42)


def run_demo():
    # =============================================================================
    # 3. DATASET VERIFICATION
    # =============================================================================
    print("\n[Demo] Initializing Datasets and Dataloaders...")

    # We set load_cached_data=False to force processing logic verification,
    # though it will save to our temp dir.
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Check one batch from train loader
    images, labels = next(iter(train_loader))

    print(f"[Demo] Train Batch Shapes - Images: {images.shape}, Labels: {labels.shape}")

    # Assertions
    # Shape: (Batch, 3, n_mels, time)
    # Time dim depends on hop_length. 16000/160 = 100 frames + 1 = 101.
    assert images.shape[0] == library.config.BATCH_SIZE, "Batch size mismatch"
    assert images.shape[1] == 3, "Channel dimension mismatch (should be 3)"
    assert images.shape[2] == library.config.N_MELS, "Mel dimension mismatch"
    assert isinstance(labels, torch.Tensor), "Labels should be a Tensor"

    print("[Demo] Dataset verification successful.")

    # =============================================================================
    # 4. MODEL VERIFICATION
    # =============================================================================
    print("\n[Demo] Initializing Model...")
    model = AudioEfficientNet(pretrained=False)  # False for speed, logic is same
    model.eval()

    # Forward pass with the batch we grabbed
    with torch.no_grad():
        outputs = model(images)

    print(f"[Demo] Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        library.config.BATCH_SIZE,
        library.config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected {(library.config.BATCH_SIZE, library.config.NUM_CLASSES)}, got {outputs.shape}"

    print("[Demo] Model verification successful.")

    # =============================================================================
    # 5. TRAINER & TRAINING LOOP VERIFICATION
    # =============================================================================
    print("\n[Demo] Initializing Trainer and starting training loop...")

    trainer = Trainer(train_loader, val_loader, test_loader)

    # Run fit (Config is set to 1 epoch)
    trainer.fit(epochs=library.config.EPOCHS)

    # Verify best model was saved
    best_model_path = os.path.join(DEMO_WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model file was not created."
    print(f"[Demo] Training finished. Best model saved at {best_model_path}")

    # =============================================================================
    # 6. INFERENCE AND SUBMISSION VERIFICATION
    # =============================================================================
    print("\n[Demo] Running Inference...")

    predictions = trainer.predict()

    print(f"[Demo] Predictions generated: {len(predictions)} samples.")
    print(f"[Demo] Sample predictions: {predictions[:5]}")

    assert len(predictions) == len(
        subset_test
    ), "Number of predictions does not match test set size."
    assert isinstance(predictions[0], str), "Predictions should be strings (labels)."

    # Save Submission
    submission_path = os.path.join(DEMO_WORKING_DIR, "submission.csv")
    save_submission(predictions, subset_test, submission_path)

    assert os.path.exists(submission_path), "Submission file not created."

    # Verify content
    df_sub = pd.read_csv(submission_path)
    assert list(df_sub.columns) == ["fname", "label"], "Submission columns mismatch."
    assert len(df_sub) == len(subset_test), "Submission length mismatch."

    print(f"[Demo] Submission saved to {submission_path}")
    print("\n[Demo] All verification steps completed successfully.")


if __name__ == "__main__":
    run_demo()
