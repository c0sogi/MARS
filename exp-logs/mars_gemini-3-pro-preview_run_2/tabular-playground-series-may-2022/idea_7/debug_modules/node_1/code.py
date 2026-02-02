import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import shutil
import warnings

# Import from the provided library
from library.config import ModelConfig, seed_everything
from library.dataset import get_dataloaders
from library.model import IIResFunnelGLU
from library.engine import train_one_epoch, validate, predict
from library.utils import save_checkpoint, load_checkpoint

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config for Speed and Isolation
    ModelConfig.WORKING_DIR = "./working/demo_run"
    ModelConfig.PROCESSED_DATA_PATH = os.path.join(
        ModelConfig.WORKING_DIR, "processed_data.npz"
    )
    ModelConfig.MODEL_SAVE_PATH = os.path.join(
        ModelConfig.WORKING_DIR, "best_model.pth"
    )
    ModelConfig.SUBMISSION_PATH = os.path.join(
        ModelConfig.WORKING_DIR, "submission.csv"
    )

    # Set hyperparams for speed
    ModelConfig.EPOCHS = 1
    ModelConfig.BATCH_SIZE = 128  # Smaller batch for debug
    ModelConfig.DEBUG = True
    ModelConfig.DEBUG_SAMPLE_SIZE = 2000  # Small subset for instant execution

    # Ensure directories exist
    ModelConfig.create_dirs()

    # Set Seed
    seed_everything(ModelConfig.SEED)
    device = torch.device(ModelConfig.DEVICE)
    print(f"    Device: {device}")
    print(f"    Working Directory: {ModelConfig.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # --------------------------------------------------------------------------
    print("\n[2] Loading and Verifying Data...")

    # We force load_cached_data=False to demonstrate processing logic,
    # or True if we want to rely on existing cache. Given the constraints,
    # we'll try to use cache if available, but the debug flag handles subsetting.
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True, debug=ModelConfig.DEBUG
    )

    # Fetch a single batch to verify shapes
    sample_batch = next(iter(train_loader))
    cont_data = sample_batch["cont"]
    cat_data = sample_batch["cat"]
    targets = sample_batch["target"]

    print(f"    Batch Size: {cont_data.size(0)}")
    print(f"    Continuous Features Shape: {cont_data.shape}")
    print(f"    Categorical Features Shape: {cat_data.shape}")
    print(f"    Targets Shape: {targets.shape}")

    # Assertions
    assert (
        cont_data.shape[1] == ModelConfig.NUM_CONT_FEATURES
    ), "Incorrect number of continuous features"
    assert (
        cat_data.shape[1] == 10
    ), "Incorrect sequence length for categorical features (f_27)"
    assert targets.dim() == 1, "Targets should be 1D tensor"

    print("    Data shapes verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # --------------------------------------------------------------------------
    print("\n[3] Initializing Model and Verifying Forward Pass...")

    model = IIResFunnelGLU().to(device)

    # Move sample batch to device
    cont_device = cont_data.to(device)
    cat_device = cat_data.to(device)

    # Forward pass
    logits = model(cont_device, cat_device)

    print(f"    Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        cont_data.size(0),
        1,
    ), f"Expected output shape {(cont_data.size(0), 1)}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "Model produced NaN logits"

    print("    Forward pass verified successfully.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n[4] Running Training Loop (1 Epoch)...")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=ModelConfig.LEARNING_RATE,
        weight_decay=ModelConfig.WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()

    # Train
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"    Training completed. Loss: {train_loss:.5f}")

    # Validate
    val_loss, val_auc = validate(model, val_loader, criterion, device)
    print(f"    Validation completed. Loss: {val_loss:.5f} | AUC: {val_auc:.5f}")

    # Assertions
    assert not np.isnan(train_loss), "Training loss is NaN"
    assert 0.0 <= val_auc <= 1.0, "AUC score out of range"

    # --------------------------------------------------------------------------
    # 5. Checkpointing Demonstration
    # --------------------------------------------------------------------------
    print("\n[5] Testing Checkpoint Mechanism...")

    # Save
    save_checkpoint(
        model,
        optimizer,
        None,
        epoch=0,
        metric=val_auc,
        filepath=ModelConfig.MODEL_SAVE_PATH,
    )
    assert os.path.exists(
        ModelConfig.MODEL_SAVE_PATH
    ), "Checkpoint file was not created"

    # Load
    # Create a new model instance to ensure loading works
    loaded_model = IIResFunnelGLU().to(device)
    checkpoint = load_checkpoint(
        ModelConfig.MODEL_SAVE_PATH, loaded_model, device=device
    )

    print(
        f"    Loaded checkpoint from epoch {checkpoint['epoch']} with metric {checkpoint['metric']:.5f}"
    )

    # Verify weights match
    model_params = dict(model.named_parameters())
    loaded_params = dict(loaded_model.named_parameters())

    for name, param in model_params.items():
        loaded_param = loaded_params[name]
        assert torch.equal(param, loaded_param), f"Parameter mismatch in {name}"

    print("    Checkpoint save/load verified.")

    # --------------------------------------------------------------------------
    # 6. Inference & Submission
    # --------------------------------------------------------------------------
    print("\n[6] Generating Submission...")

    # Predict
    predictions = predict(loaded_model, test_loader, device)

    print(f"    Predictions generated. Count: {len(predictions)}")
    print(f"    Sample predictions: {predictions[:5]}")

    # Assertions
    assert len(predictions) == len(
        test_ids
    ), "Mismatch between predictions and test IDs"
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    ), "Predictions out of probability range [0, 1]"

    # Create Submission DataFrame
    submission = pd.DataFrame({"id": test_ids, "target": predictions})

    # Save
    submission.to_csv(ModelConfig.SUBMISSION_PATH, index=False)
    print(f"    Submission saved to {ModelConfig.SUBMISSION_PATH}")

    # Verify File
    df_check = pd.read_csv(ModelConfig.SUBMISSION_PATH)
    assert df_check.shape == (len(test_ids), 2), "Submission file has incorrect shape"
    assert list(df_check.columns) == [
        "id",
        "target",
    ], "Submission file has incorrect columns"

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    run_demo()
