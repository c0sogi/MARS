import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import prepare_data, WhaleDataset
from library.model import DualStreamEfficientNet
from library.trainer import Trainer


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("\n=== Setting up Demo Configuration ===")

    class DemoConfig(Config):
        """
        Configuration overrides for a fast demonstration run.
        """

        WORKING_DIR = "./working/demo_run"
        SUBMISSION_PATH = "./working/demo_run/submission/submission.csv"

        # Reduce training duration
        EPOCHS = 2
        BATCH_SIZE = 8
        PATIENCE = 2

        # Disable multiprocessing for small data demo to avoid overhead
        NUM_WORKERS = 0

        # Ensure deterministic behavior
        SEED = 42

    # Initialize config and directories
    config = DemoConfig()
    DemoConfig.setup()
    seed_everything(config.SEED)

    device = torch.device(config.DEVICE)
    logger = get_logger("DemoRunner")
    logger.info(f"Device: {device}")
    logger.info(f"Working Directory: {config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Preparation (Subset)
    # -------------------------------------------------------------------------
    print("\n=== Step 2: Preparing Data Subsets ===")

    # Load full metadata
    train_full_df = pd.read_csv(os.path.join(config.METADATA_DIR, "train.csv"))
    val_full_df = pd.read_csv(os.path.join(config.METADATA_DIR, "val.csv"))

    # Create tiny subsets for demonstration (20 train, 10 val)
    train_subset_df = train_full_df.head(20).copy()
    val_subset_df = val_full_df.head(10).copy()

    logger.info(f"Training subset size: {len(train_subset_df)}")
    logger.info(f"Validation subset size: {len(val_subset_df)}")

    # Process data using library function
    # Note: prepare_data handles caching. We use specific cache names for this demo.
    logger.info("Processing training subset...")
    X1_train, X2_train, Y_train, _ = prepare_data(
        train_subset_df, config, cache_name="train_debug", load_cached_data=False
    )

    logger.info("Processing validation subset...")
    X1_val, X2_val, Y_val, _ = prepare_data(
        val_subset_df, config, cache_name="val_debug", load_cached_data=False
    )

    # Verification: Check Data Shapes
    # Expected: (N, 1, F, T)
    # Stream 1: 384 Mels, ~40 frames (2.0s / 0.05s hop) -> Actual T depends on padding/stft
    # Stream 2: 128 Mels, ~200 frames (2.0s / 0.01s hop)
    logger.info(f"X1 Train Shape: {X1_train.shape}")
    logger.info(f"X2 Train Shape: {X2_train.shape}")

    assert len(X1_train) == 20, "Incorrect number of training samples processed"
    assert X1_train.ndim == 4 and X1_train.shape[1] == 1, "X1 should be (N, 1, F, T)"
    assert X2_train.ndim == 4 and X2_train.shape[1] == 1, "X2 should be (N, 1, F, T)"
    assert len(Y_train) == 20, "Incorrect number of labels"

    print("Data preparation verification passed.")

    # -------------------------------------------------------------------------
    # 3. Dataset & DataLoader
    # -------------------------------------------------------------------------
    print("\n=== Step 3: Verifying Dataset and DataLoader ===")

    train_dataset = WhaleDataset(X1_train, X2_train, Y_train, augment=True)
    val_dataset = WhaleDataset(X1_val, X2_val, Y_val, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # Fetch one batch to verify
    x1_batch, x2_batch, y_batch = next(iter(train_loader))

    logger.info(f"Batch X1: {x1_batch.shape}")
    logger.info(f"Batch X2: {x2_batch.shape}")
    logger.info(f"Batch Y: {y_batch.shape}")

    assert x1_batch.shape[0] == config.BATCH_SIZE or x1_batch.shape[0] == len(
        train_subset_df
    ), "Batch size mismatch"
    assert torch.is_tensor(x1_batch), "Output should be a tensor"
    assert x1_batch.dtype == torch.float32, "Data type should be float32"

    print("Dataset/Loader verification passed.")

    # -------------------------------------------------------------------------
    # 4. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n=== Step 4: Model Initialization and Forward Pass ===")

    model = DualStreamEfficientNet(config).to(device)

    # Move batch to device
    x1_batch = x1_batch.to(device)
    x2_batch = x2_batch.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(x1_batch, x2_batch)

    logger.info(f"Model Output Shape: {output.shape}")

    assert output.shape == (x1_batch.size(0), 1), "Output shape should be (B, 1)"
    assert not torch.isnan(output).any(), "Model produced NaN values"

    print("Model forward pass verification passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n=== Step 5: Executing Training Loop ===")

    # Setup training components
    pos_weight = torch.tensor([config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
    )

    # Run training
    best_model_path = trainer.fit()

    # Verify artifact creation
    assert os.path.exists(best_model_path), "Best model file was not created"
    print(f"Training loop completed successfully. Model saved to {best_model_path}")

    # -------------------------------------------------------------------------
    # 6. Inference Simulation
    # -------------------------------------------------------------------------
    print("\n=== Step 6: Inference Verification ===")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for x1, x2, _ in val_loader:
            x1, x2 = x1.to(device), x2.to(device)
            out = model(x1, x2)
            prob = torch.sigmoid(out).cpu().numpy()
            preds.extend(prob)

    preds = np.array(preds).flatten()

    logger.info(f"Predictions: {preds[:5]}...")

    assert len(preds) == len(
        val_subset_df
    ), "Number of predictions does not match validation set size"
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions must be probabilities [0, 1]"

    print("Inference verification passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
