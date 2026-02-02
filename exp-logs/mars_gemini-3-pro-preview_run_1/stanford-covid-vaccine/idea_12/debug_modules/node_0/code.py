import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.utils import set_seed, mcrmse, format_submission
from library.data import get_dataloaders
from library.model import InteractionAwareModel
from library.train import MaskedMSELoss, train_one_epoch, validate


def run_demo():
    print("=== Starting RNA Degradation Prediction Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    # Modify Config to run a lightweight version for demonstration purposes
    print("[1/6] Configuring parameters for demo run...")

    # Paths
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Model Architecture (Tiny version for speed)
    Config.HIDDEN_DIM = 64
    Config.NUM_LAYERS = 2
    Config.EMBED_DIM_SEQ = 16
    Config.EMBED_DIM_LOOP = 16
    Config.EMBED_DIM_BOND = 16
    Config.EMBED_DIM_DISTANCE = 16
    # Important: Re-calculate INPUT_DIM as it is a derived property in Config
    Config.INPUT_DIM = (
        Config.EMBED_DIM_SEQ
        + Config.EMBED_DIM_LOOP
        + Config.EMBED_DIM_BOND
        + Config.EMBED_DIM_DISTANCE
    )

    # Training Hyperparameters
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 32
    Config.DEVICE = "cpu"  # Use CPU for deterministic demo execution
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    # Create working directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set reproducibility
    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("[2/6] Loading and processing data...")
    # get_dataloaders handles caching automatically.
    # Since we changed CACHE_DIR, this will trigger a fresh process from metadata.
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Verification: Check data shapes
    print("      Verifying data integrity...")
    sample_inputs, sample_targets = next(iter(train_loader))

    # Check Input Shapes
    # Seq: (Batch, 107)
    assert sample_inputs["seq"].shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Incorrect sequence shape: {sample_inputs['seq'].shape}"
    # Dist: (Batch, 107, Embed_Dim_Dist)
    assert sample_inputs["dist"].shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.EMBED_DIM_DISTANCE,
    ), f"Incorrect distance feature shape: {sample_inputs['dist'].shape}"

    # Check Target Shapes
    # Targets: (Batch, 68, 3) -> Reactivity, deg_Mg_pH10, deg_Mg_50C
    assert sample_targets.shape == (
        Config.BATCH_SIZE,
        Config.PRED_LEN,
        Config.NUM_TARGETS,
    ), f"Incorrect target shape: {sample_targets.shape}"

    print(f"      Data loaded successfully. Train batches: {len(train_loader)}")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("[3/6] Initializing model...")
    device = torch.device(Config.DEVICE)
    model = InteractionAwareModel().to(device)

    # Verification: Forward pass
    with torch.no_grad():
        # Move sample inputs to device
        inputs_device = {k: v.to(device) for k, v in sample_inputs.items()}
        outputs = model(inputs_device)

        # Output shape should be (Batch, 107, 3)
        assert outputs.shape == (
            Config.BATCH_SIZE,
            Config.SEQ_LEN,
            Config.NUM_TARGETS,
        ), f"Incorrect model output shape: {outputs.shape}"

    print("      Model initialized and forward pass verified.")

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print(f"[4/6] Starting training for {Config.EPOCHS} epochs...")

    criterion = MaskedMSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    for epoch in range(Config.EPOCHS):
        # Use the provided train_one_epoch function
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        print(f"      Epoch {epoch+1}: Avg Loss = {train_loss:.6f}")

    # -------------------------------------------------------------------------
    # 5. Validation
    # -------------------------------------------------------------------------
    print("[5/6] Validating model...")
    # Use the provided validate function
    val_score = validate(model, val_loader, device)
    print(f"      Validation MCRMSE: {val_score:.6f}")

    # Save the demo model
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print(f"      Model saved to {Config.MODEL_SAVE_PATH}")

    # -------------------------------------------------------------------------
    # 6. Inference & Submission
    # -------------------------------------------------------------------------
    print("[6/6] Generating submission...")

    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for inputs, ids in test_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(inputs)  # (Batch, 107, 3)

            all_preds.append(outputs.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate predictions
    predictions = np.concatenate(all_preds, axis=0)

    # Verification: Check prediction count matches test set size
    assert len(all_ids) == 240, f"Expected 240 test samples, got {len(all_ids)}"
    assert predictions.shape == (240, Config.SEQ_LEN, Config.NUM_TARGETS)

    # Format and save submission using library utility
    format_submission(all_ids, predictions, save_path=Config.SUBMISSION_PATH)

    # Final Verification of the output file
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"      Submission file created: {Config.SUBMISSION_PATH}")
        print(f"      Submission shape: {df_sub.shape}")

        # Expected rows: 240 samples * 107 positions = 25680
        expected_rows = 240 * 107
        assert (
            len(df_sub) == expected_rows
        ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
