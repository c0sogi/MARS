import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.data_processor import DataProcessor
from library.dataset import ManufacturingDataset
from library.model import GUTClassifier
from library.engine import train_model, generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup and Configuration
    set_seed(42)
    print("Initializing Configuration (Debug Mode)...")

    # Initialize Config with debug=True to use a subset of data (5000 samples) and fewer epochs (2)
    config = Config(debug=True)

    # Override working directory for this demo to isolate outputs
    config.working_dir = "./working/demo_execution"
    os.makedirs(config.working_dir, exist_ok=True)

    # Ensure submission path is also handled cleanly
    config.submission_dir = os.path.join(config.working_dir, "submission")
    os.makedirs(config.submission_dir, exist_ok=True)
    config.submission_path = os.path.join(config.submission_dir, "submission.csv")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Processing
    print("Running Data Processor...")
    processor = DataProcessor(config)

    # Force processing from scratch (load_cached_data=False) to demonstrate pipeline logic
    data = processor.process_data(load_cached_data=False)

    # --- Verification: Data Shapes & Integrity ---
    print("Verifying processed data...")
    n_train_samples = data["X_num_train"].shape[0]
    n_num_features = len(config.numerical_features)
    n_seq_len = config.sequence_len

    # Check dimensions
    assert data["X_num_train"].shape == (
        n_train_samples,
        n_num_features,
    ), f"Mismatch in X_num_train shape: {data['X_num_train'].shape}"
    assert data["X_seq_train"].shape == (
        n_train_samples,
        n_seq_len,
    ), f"Mismatch in X_seq_train shape: {data['X_seq_train'].shape}"

    # Check for NaNs
    assert not np.isnan(
        data["X_num_train"]
    ).any(), "NaNs detected in numerical training data"
    assert not np.isnan(data["y_train"]).any(), "NaNs detected in training targets"
    print("Data verification passed.")

    # 3. Dataset and DataLoader Creation
    print("Creating Datasets and Loaders...")
    train_dataset = ManufacturingDataset(
        data["X_num_train"], data["X_seq_train"], data["y_train"]
    )
    val_dataset = ManufacturingDataset(
        data["X_num_val"], data["X_seq_val"], data["y_val"]
    )
    test_dataset = ManufacturingDataset(data["X_num_test"], data["X_seq_test"], None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for simple script execution
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )

    # --- Verification: DataLoader Batch Structure ---
    print("Verifying DataLoader batch...")
    sample_batch = next(iter(train_loader))
    assert (
        "x_num" in sample_batch and "x_seq" in sample_batch and "target" in sample_batch
    )
    assert sample_batch["x_num"].shape[1] == n_num_features
    assert sample_batch["x_seq"].shape[1] == n_seq_len
    print("DataLoader verification passed.")

    # 4. Model Initialization
    print("Initializing GUTClassifier...")
    model = GUTClassifier(config).to(device)

    # --- Verification: Model Forward Pass ---
    print("Verifying model forward pass...")
    model.eval()
    with torch.no_grad():
        dummy_x_num = sample_batch["x_num"].to(device)
        dummy_x_seq = sample_batch["x_seq"].to(device)
        logits = model(dummy_x_num, dummy_x_seq)

        # Expect output shape (Batch_Size, 1)
        assert logits.shape == (
            dummy_x_num.size(0),
            1,
        ), f"Model output shape mismatch. Expected {(dummy_x_num.size(0), 1)}, got {logits.shape}"
    print("Model verification passed.")

    # 5. Training Loop
    print("Starting Training...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Calculate steps for OneCycleLR
    total_steps = config.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=config.pct_start,
        div_factor=config.div_factor,
        final_div_factor=config.final_div_factor,
    )

    # Execute training using the engine
    best_auc = train_model(
        model, train_loader, val_loader, optimizer, scheduler, criterion, device, config
    )

    # --- Verification: Training Result ---
    assert 0.0 <= best_auc <= 1.0, f"Invalid AUC score: {best_auc}"
    print(f"Training completed successfully. Best AUC: {best_auc:.4f}")

    # 6. Submission Generation
    print("Generating Submission...")
    generate_submission(model, test_loader, data["ids_test"], device, config)

    # --- Verification: Submission File ---
    print("Verifying submission file...")
    if not os.path.exists(config.submission_path):
        raise FileNotFoundError(
            f"Submission file not found at {config.submission_path}"
        )

    df_sub = pd.read_csv(config.submission_path)
    expected_rows = len(data["ids_test"])
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"
    assert list(df_sub.columns) == [
        config.id_col,
        config.target_col,
    ], "Submission columns mismatch"
    print("Submission verification passed.")
    print("Demo execution completed successfully.")


if __name__ == "__main__":
    main()
