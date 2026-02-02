import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, get_optimizer_params
from library.dataset import get_datasets, ManufacturingDataset
from library.model import HybridSwiGLUNet
from library.engine import fit, generate_submission

if __name__ == "__main__":
    print("--- Starting Demonstration Script ---")

    # --------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # --------------------------------------------------------------------------
    # Modify Config to run a lightweight version of the model for demo purposes
    print("Configuring lightweight hyperparameters...")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 1024

    # Reduce Model Complexity
    Config.EMBED_DIM = 16
    Config.TRANSFORMER_LAYERS = 1
    Config.TRANSFORMER_HEADS = 2
    Config.BACKBONE_STAGES = [64, 32]  # Smaller backbone

    # Ensure directories exist
    Config.setup()

    # --------------------------------------------------------------------------
    # 2. Initialization
    # --------------------------------------------------------------------------
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # --------------------------------------------------------------------------
    # 3. Data Loading & Subsetting
    # --------------------------------------------------------------------------
    print("Loading datasets...")
    # Load full datasets (cached or fresh)
    train_ds_full, val_ds_full, test_ds_full = get_datasets(load_cached_data=True)

    # Create subsets for rapid demonstration
    # We take a small slice of training/validation data to ensure the 'fit' function finishes fast
    SUBSET_SIZE_TRAIN = 5000
    SUBSET_SIZE_VAL = 1000

    print(f"Subsetting training data to {SUBSET_SIZE_TRAIN} samples for demo...")
    train_ds = ManufacturingDataset(
        continuous=train_ds_full.continuous[:SUBSET_SIZE_TRAIN],
        categorical=train_ds_full.categorical[:SUBSET_SIZE_TRAIN],
        targets=train_ds_full.targets[:SUBSET_SIZE_TRAIN],
    )

    val_ds = ManufacturingDataset(
        continuous=val_ds_full.continuous[:SUBSET_SIZE_VAL],
        categorical=val_ds_full.categorical[:SUBSET_SIZE_VAL],
        targets=val_ds_full.targets[:SUBSET_SIZE_VAL],
    )

    # We keep the full test set to generate a valid submission file structure
    test_ds = test_ds_full

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --------------------------------------------------------------------------
    # 4. Model Instantiation & Logic Verification
    # --------------------------------------------------------------------------
    print("Initializing model...")
    model = HybridSwiGLUNet().to(device)

    # Verify Forward Pass Logic
    print("Verifying model forward pass...")
    dummy_cont = torch.randn(10, Config.NUM_CONTINUOUS_FEATURES).to(device)
    dummy_cat = torch.randint(0, Config.VOCAB_SIZE, (10, Config.SEQUENCE_LENGTH)).to(
        device
    )

    with torch.no_grad():
        dummy_out = model(dummy_cont, dummy_cat)

    # Assert output shape is (Batch, 1)
    assert dummy_out.shape == (
        10,
        1,
    ), f"Expected output shape (10, 1), got {dummy_out.shape}"
    print("Model verification passed.")

    # --------------------------------------------------------------------------
    # 5. Training Setup
    # --------------------------------------------------------------------------
    optimizer_params = get_optimizer_params(model)
    optimizer = torch.optim.AdamW(optimizer_params)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    # --------------------------------------------------------------------------
    # 6. Execution (Training)
    # --------------------------------------------------------------------------
    print("Starting training loop...")
    best_auc = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        patience=1,  # Strict patience for demo
    )

    print(f"Demo training finished. Best AUC: {best_auc:.4f}")

    # --------------------------------------------------------------------------
    # 7. Inference & Submission
    # --------------------------------------------------------------------------
    # We need the test IDs. The dataset object doesn't store IDs directly in the tensor properties,
    # but preprocess_data returns them. Since we used get_datasets, we need to reload the IDs
    # or rely on the fact that test_ds_full corresponds to the metadata order.
    # The safest way using provided library is to load the test metadata or cached IDs.

    # Load cached IDs directly to ensure alignment
    data_cache = np.load(Config.CACHE_PATH)
    test_ids = data_cache["test_ids"]

    # Verify alignment
    assert len(test_ids) == len(
        test_ds
    ), "Mismatch between Test IDs and Test Dataset length."

    generate_submission(
        model=model, dataloader=test_loader, test_ids=test_ids, device=device
    )

    # --------------------------------------------------------------------------
    # 8. Final Validation
    # --------------------------------------------------------------------------
    print("Validating submission file...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not generated at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check shape
    assert df_sub.shape == (
        100000,
        2,
    ), f"Submission shape mismatch. Expected (100000, 2), got {df_sub.shape}"

    # Check columns
    assert list(df_sub.columns) == ["id", "target"], "Submission columns mismatch."

    # Check ID alignment
    assert df_sub["id"].iloc[0] == 900000, "First ID in submission is incorrect."

    # Check probability range
    assert (
        df_sub["target"].min() >= 0.0 and df_sub["target"].max() <= 1.0
    ), "Predictions out of probability range [0, 1]."

    print("--- Demonstration Complete: All checks passed. ---")
