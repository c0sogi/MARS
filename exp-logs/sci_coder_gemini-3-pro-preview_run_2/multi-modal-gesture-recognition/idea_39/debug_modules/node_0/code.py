import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. Configuration Patching (Must be done before importing dependent modules)
# -----------------------------------------------------------------------------
import library.config

# Define a patched hyperparameter function for the demo
original_get_hp = library.config.get_hyperparams


def patched_get_hyperparams(debug=True):
    # Force debug=True to get the base debug config
    hp = original_get_hp(debug=True)

    # Override specific parameters for the demo run
    hp["epochs"] = 1  # Run only 1 epoch
    hp["batch_size"] = 2  # Small batch size
    hp["sample_size"] = 10  # Use only 10 samples for speed
    hp["num_workers"] = 0  # Avoid multiprocessing overhead for small data
    hp["patience"] = 1  # minimal patience
    hp["lstm_hidden_size"] = 32  # Reduce model size for speed
    hp["tcn_channels"] = 32  # Reduce model size for speed
    hp["tcn_layers"] = 2  # Reduce depth for speed
    return hp


# Apply the patch
library.config.get_hyperparams = patched_get_hyperparams

# Redirect paths to a demo directory in ./working
demo_dir = os.path.join(os.getcwd(), "working", "demo_run")
os.makedirs(demo_dir, exist_ok=True)

# Update the global PATHS dictionary in the config module
library.config.PATHS["working"] = demo_dir
library.config.PATHS["checkpoints"] = os.path.join(demo_dir, "checkpoints")
library.config.PATHS["cache"] = os.path.join(demo_dir, "cache")
library.config.PATHS["predictions"] = os.path.join(demo_dir, "predictions")
library.config.PATHS["submission"] = os.path.join(
    demo_dir, "submission", "submission.csv"
)
library.config.PATHS["model_save_path"] = os.path.join(
    demo_dir, "checkpoints", "best_model.pth"
)

# Ensure subdirectories exist
os.makedirs(library.config.PATHS["checkpoints"], exist_ok=True)
os.makedirs(library.config.PATHS["cache"], exist_ok=True)
os.makedirs(library.config.PATHS["predictions"], exist_ok=True)
os.makedirs(os.path.dirname(library.config.PATHS["submission"]), exist_ok=True)

print(f"Configuration patched. Working directory: {demo_dir}")

# -----------------------------------------------------------------------------
# 2. Import Modules (Now that config is patched)
# -----------------------------------------------------------------------------
from library.utils import set_seed, compute_levenshtein
from library.data_loader import get_dataloaders
from library.model import MG_CRGN
from library.losses import ActionSegmentationLoss
from library.trainer import Trainer
from library.inference import run_inference

if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)
    print("\n--- 1. Testing Utilities ---")

    # Verify Levenshtein distance
    seq1 = [1, 2, 3]
    seq2 = [1, 3]
    dist = compute_levenshtein(seq1, seq2)
    print(f"Levenshtein distance between {seq1} and {seq2}: {dist}")
    assert dist == 1, "Levenshtein distance calculation is incorrect."
    print("Utility verification passed.")

    print("\n--- 2. Testing Data Loading ---")
    # Load data loaders (will use sample_size=10 from patched config)
    # We disable cache loading to ensure the data processing logic runs
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch one batch
    batch = next(iter(train_loader))
    features, targets, mask, sample_ids = batch

    print(f"Batch Sample IDs: {sample_ids}")
    print(f"Features Shape: {features.shape} (B, C, T)")
    print(f"Targets Shape: {targets.shape} (B, T)")
    print(f"Mask Shape: {mask.shape} (B, T)")

    # Assertions
    # Features should be (B, 85, T) -> 36 Pos + 36 Vel + 13 Audio = 85
    assert (
        features.shape[1] == 85
    ), f"Expected 85 feature channels, got {features.shape[1]}"
    assert features.shape[0] == 2, f"Expected batch size 2, got {features.shape[0]}"
    assert (
        features.shape[2] == targets.shape[1]
    ), "Temporal dimension mismatch between features and targets"
    print("Data loading verification passed.")

    print("\n--- 3. Testing Model and Loss ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MG_CRGN().to(device)
    criterion = ActionSegmentationLoss().to(device)

    # Move batch to device
    features = features.to(device)
    targets = targets.to(device)
    mask = mask.to(device)

    # Forward pass
    outputs = model(features, mask)

    # Model returns a list of outputs for deep supervision (3 stages)
    assert isinstance(outputs, list), "Model output should be a list"
    assert len(outputs) == 3, f"Expected 3 stage outputs, got {len(outputs)}"

    # Check shape of last stage output: (B, NumClasses+1, T)
    # NumClasses = 21 (0-20), so channels should be 22
    last_stage_out = outputs[-1]
    print(f"Model Output Shape (Stage 3): {last_stage_out.shape}")
    assert (
        last_stage_out.shape[1] == 22
    ), f"Expected 22 output channels (21 classes + 1 boundary), got {last_stage_out.shape[1]}"
    assert (
        last_stage_out.shape[2] == features.shape[2]
    ), "Output temporal dimension mismatch"

    # Compute Loss
    loss = criterion(outputs, targets, mask)
    print(f"Computed Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    print("Model and Loss verification passed.")

    print("\n--- 4. Running Training Loop (Trainer) ---")
    # Initialize Trainer
    trainer = Trainer()

    # Run training (1 epoch, small subset)
    trainer.fit()

    # Verify checkpoint creation
    checkpoint_path = library.config.PATHS["model_save_path"]
    assert os.path.exists(
        checkpoint_path
    ), f"Checkpoint file not found at {checkpoint_path}"
    print(f"Training complete. Checkpoint saved at {checkpoint_path}")

    print("\n--- 5. Running Inference ---")
    # Run inference pipeline
    # This will load the model we just trained and generate predictions on the test subset
    run_inference(load_cached_data=False)

    # Verify submission file
    submission_path = library.config.PATHS["submission"]
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    # Check content
    df_sub = pd.read_csv(submission_path, header=None)
    print(f"Submission generated with {len(df_sub)} rows.")
    # We used sample_size=10 for test set as well, so we expect roughly 10 rows
    # (or fewer if some samples failed loading, but usually 10)
    assert len(df_sub) > 0, "Submission file is empty"

    print("Inference verification passed.")
    print("\nAll demonstrations completed successfully.")
