import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, MCRMSEMetric
from library.loss import AnchoredMCRMSELoss
from library.data import load_data
from library.model import ADFRN
from library.train import train_one_epoch, validate, generate_submission

if __name__ == "__main__":
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print(">>> Setting up configuration for demo...")

    # Override Config for a fast, mini demonstration
    Config.SUBSET_SIZE = 50  # Only use 50 samples
    Config.BATCH_SIZE = 4  # Small batch size
    Config.EPOCHS = 2  # Run 2 epochs
    Config.WORKING_DIR = "./working/demo_execution"
    Config.TRAIN_CACHE_FILE = "train_data_demo.npz"
    Config.VAL_CACHE_FILE = "val_data_demo.npz"
    Config.TEST_CACHE_FILE = "test_data_demo.npz"

    # Ensure clean working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # =========================================================================
    # 2. Data Loading Demonstration
    # =========================================================================
    print("\n>>> Loading Data...")

    # Load Train Data (Force reload to ignore existing cache for this demo)
    train_loader = load_data(mode="train", load_cached_data=False)
    val_loader = load_data(mode="val", load_cached_data=False)
    test_loader = load_data(mode="test", load_cached_data=False)

    # Verify Train Loader
    print(f"Train batches: {len(train_loader)}")
    batch_inputs, batch_partners, batch_targets = next(iter(train_loader))

    # Assert Shapes
    # Inputs: (B, 107, 18) -> Sequence(4)+Struct(3)+Loop(7)+Partner(4)
    assert batch_inputs.shape == (
        Config.BATCH_SIZE,
        107,
        18,
    ), f"Unexpected input shape: {batch_inputs.shape}"
    # Partner Indices: (B, 107)
    assert batch_partners.shape == (
        Config.BATCH_SIZE,
        107,
    ), f"Unexpected partner shape: {batch_partners.shape}"
    # Targets: (B, 107, 5)
    assert batch_targets.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), f"Unexpected target shape: {batch_targets.shape}"

    print("Data loading verified successfully.")

    # =========================================================================
    # 3. Model Initialization & Forward Pass
    # =========================================================================
    print("\n>>> Initializing Model...")
    model = ADFRN().to(device)

    # Move batch to device
    b_inputs = batch_inputs.to(device)
    b_partners = batch_partners.to(device)
    b_targets = batch_targets.to(device)

    print("Running forward pass...")
    # The model returns two outputs: Pass 1 (no feedback) and Pass 2 (feedback)
    y1, y2 = model(b_inputs, b_partners)

    # Verify Output Shapes
    assert y1.shape == (Config.BATCH_SIZE, 107, 5), f"Pass 1 shape mismatch: {y1.shape}"
    assert y2.shape == (Config.BATCH_SIZE, 107, 5), f"Pass 2 shape mismatch: {y2.shape}"

    print("Forward pass successful.")

    # =========================================================================
    # 4. Loss & Metric Demonstration
    # =========================================================================
    print("\n>>> Testing Loss and Metric...")
    criterion = AnchoredMCRMSELoss()
    metric = MCRMSEMetric()

    # Calculate Loss
    loss = criterion(y1, y2, b_targets)
    print(f"Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    # Calculate Metric
    metric.update(y2, b_targets)
    score = metric.compute()
    print(f"Calculated MCRMSE Score: {score:.4f}")

    assert score >= 0, "Metric score should be non-negative"

    # Reset metric verification
    metric.reset()
    assert metric.total_count == 0, "Metric reset failed"

    # =========================================================================
    # 5. Training Loop Demonstration
    # =========================================================================
    print("\n>>> Running Mini-Training Loop...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(1, Config.EPOCHS + 1):
        # Train Step
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validation Step
        val_loss, val_mcrmse = validate(model, val_loader, criterion, metric, device)

        print(
            f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val MCRMSE={val_mcrmse:.4f}"
        )

        # Basic assertions to ensure values are changing/valid
        assert train_loss > 0
        assert val_loss > 0

    # Save the "best" model for the submission step
    model_path = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # =========================================================================
    # 6. Submission Generation
    # =========================================================================
    print("\n>>> Generating Submission...")

    submission_file = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Reload model to verify state dict loading works
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )

    # Generate submission
    generate_submission(model, device, submission_file)

    # Verify Submission File
    assert os.path.exists(submission_file), "Submission file was not created"

    sub_df = pd.read_csv(submission_file)
    print(f"Submission shape: {sub_df.shape}")
    print(f"Submission columns: {sub_df.columns.tolist()}")

    # Expected rows: Config.SUBSET_SIZE (test set) * 107 (seq len)
    # Note: Config.SUBSET_SIZE applies to load_data.
    # Since we used load_data('test') with SUBSET_SIZE=50, we expect 50 * 107 rows.
    expected_rows = Config.SUBSET_SIZE * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"

    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert list(sub_df.columns) == expected_cols, "Column mismatch in submission"

    print("\n>>> Demo Completed Successfully!")
