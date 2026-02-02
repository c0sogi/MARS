import sys
import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.optim as optim

# Add current directory to path to ensure library imports work correctly
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, get_device
from library.features import StructuralFeatureGenerator, generate_all_features
from library.dataset import get_dataloaders
from library.model import SiameseDebertaGated
from library.engine import train_model, predict


def run_pipeline_demo():
    print("================================================================")
    print("   Chatbot Preference Prediction: End-to-End Pipeline Demo      ")
    print("================================================================")

    # ------------------------------------------------------------------
    # 1. Configuration Setup
    # ------------------------------------------------------------------
    print("\n[1] Configuring Environment for Rapid Execution...")

    # Override Config for speed and debugging
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 10  # Use only 10 rows for this demo
    Config.NUM_EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 2
    Config.VALID_BATCH_SIZE = 2
    Config.GRAD_ACCUM_STEPS = 1

    # Define a clean working directory for this run
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.MODEL_OUTPUT_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_FILE_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up previous run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Create directories
    Config.setup()

    # Set reproducibility
    seed_everything(Config.SEED)
    device = get_device()

    print(f"    Device: {device}")
    print(f"    Working Dir: {Config.WORKING_DIR}")
    print(f"    Batch Size: {Config.TRAIN_BATCH_SIZE}")

    # ------------------------------------------------------------------
    # 2. Feature Generation
    # ------------------------------------------------------------------
    print("\n[2] Generating Structural Features...")
    # This reads the metadata CSVs, computes features, and saves to cache.
    # Note: This processes the full CSVs first, then slicing happens in the dataset loader.
    train_feats, val_feats, test_feats = generate_all_features(load_cached_data=False)

    # Verify shapes (Note: These are full dataset shapes before DEBUG slicing)
    print(f"    Full Train Features Shape: {train_feats.shape}")
    assert train_feats.shape[1] == 6, "Structural features must have 6 columns"
    assert not np.isnan(train_feats).any(), "Features contain NaNs"
    print("    Feature generation verified.")

    # ------------------------------------------------------------------
    # 3. Data Loading
    # ------------------------------------------------------------------
    print("\n[3] Initializing DataLoaders...")
    # This loads features from cache and applies the DEBUG_SUBSET_SIZE slice
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_features=True)

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))
    print("    Inspecting one training batch:")
    for k, v in batch.items():
        print(f"      - {k}: {v.shape}")

    # Assertions
    assert "input_ids_a" in batch
    assert "input_ids_b" in batch
    assert "structural_features" in batch
    assert "labels" in batch
    assert batch["input_ids_a"].shape[0] == Config.TRAIN_BATCH_SIZE
    assert batch["structural_features"].shape[1] == 6
    assert batch["labels"].shape[1] == 3
    print("    DataLoaders verified.")

    # ------------------------------------------------------------------
    # 4. Model Initialization
    # ------------------------------------------------------------------
    print("\n[4] Initializing Model...")
    model = SiameseDebertaGated()
    model.to(device)

    # Run a dummy forward pass to check architecture
    print("    Running dummy forward pass...")
    model.eval()
    with torch.no_grad():
        ids_a = batch["input_ids_a"].to(device)
        mask_a = batch["attention_mask_a"].to(device)
        ids_b = batch["input_ids_b"].to(device)
        mask_b = batch["attention_mask_b"].to(device)
        struct = batch["structural_features"].to(device)

        outputs = model(ids_a, mask_a, ids_b, mask_b, struct)

    print(f"    Output Shape: {outputs.shape}")
    assert outputs.shape == (Config.TRAIN_BATCH_SIZE, 3), "Output shape mismatch"
    print("    Model initialization verified.")

    # ------------------------------------------------------------------
    # 5. Training Loop
    # ------------------------------------------------------------------
    print("\n[5] Executing Training Loop...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    # Train for 1 epoch
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        num_epochs=Config.NUM_EPOCHS,
        patience=1,
    )

    assert os.path.exists(Config.MODEL_OUTPUT_PATH), "Model checkpoint not found"
    print(f"    Model saved to {Config.MODEL_OUTPUT_PATH}")

    # ------------------------------------------------------------------
    # 6. Prediction
    # ------------------------------------------------------------------
    print("\n[6] Running Prediction on Test Set...")
    predict(trained_model, test_loader, device)

    assert os.path.exists(Config.SUBMISSION_FILE_PATH), "Submission file not found"

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_FILE_PATH)
    print("    Submission Head:")
    print(sub_df.head())

    assert len(sub_df) == Config.DEBUG_SUBSET_SIZE, "Submission length mismatch"
    assert list(sub_df.columns) == [
        "id",
        "winner_model_a",
        "winner_model_b",
        "winner_tie",
    ]

    # Check probability constraints
    row_sums = sub_df[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    print("\n================================================================")
    print("   Pipeline Demonstration Completed Successfully")
    print("================================================================")


if __name__ == "__main__":
    run_pipeline_demo()
