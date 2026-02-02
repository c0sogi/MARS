import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.utils import seed_everything
from library.dataset import INatDataset, get_transforms
from library.model import get_model, get_criterion, get_optimizer, get_scheduler
from library.engine import fit, generate_predictions


def main():
    # 1. Setup and Configuration
    print("Setting up configuration and seeding...")
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define paths
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/demo_run"
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    os.makedirs(WORKING_DIR, exist_ok=True)

    # Hyperparameters for the demo (Optimized for speed)
    BATCH_SIZE = 8
    NUM_CLASSES = 1010
    EPOCHS = 1
    DEBUG_LIMIT = 5  # Limit to 5 batches per epoch for demonstration speed
    SUBSET_SIZE = 100  # Limit dataset size for demonstration

    # 2. Data Loading
    print("Loading metadata...")
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))

    # Subset data for rapid demonstration
    train_df = train_df.head(SUBSET_SIZE)
    val_df = val_df.head(SUBSET_SIZE)

    print(f"Train subset size: {len(train_df)}")
    print(f"Val subset size: {len(val_df)}")

    # Instantiate Datasets
    train_dataset = INatDataset(
        train_df, transform=get_transforms("train"), is_test=False
    )
    val_dataset = INatDataset(val_df, transform=get_transforms("val"), is_test=False)

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Verification: Check data loading
    sample_img, sample_target = next(iter(train_loader))
    assert sample_img.dim() == 4, "Image batch should be 4-dimensional (B, C, H, W)"
    assert sample_img.shape[1] == 3, "Images should have 3 channels"
    assert sample_target.shape[0] == BATCH_SIZE, "Target batch size mismatch"
    print("Data loading verification passed.")

    # 3. Model Initialization
    print("Initializing model...")
    model = get_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # Verification: Check model output shape
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, 256, 256).to(device)
        dummy_output = model(dummy_input)
        assert dummy_output.shape == (
            2,
            NUM_CLASSES,
        ), f"Model output shape mismatch. Expected (2, {NUM_CLASSES}), got {dummy_output.shape}"
    print("Model initialization verification passed.")

    # 4. Training Setup
    criterion = get_criterion()
    optimizer = get_optimizer(model, learning_rate=1e-4)
    scheduler = get_scheduler(optimizer, T_max=EPOCHS)

    # 5. Training Loop (using engine.fit)
    print("Starting training loop (demo)...")
    model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=EPOCHS,
        checkpoint_path=CHECKPOINT_PATH,
        patience=1,
        debug_limit=DEBUG_LIMIT,  # Limits steps per epoch for speed
    )

    # Verification: Check if checkpoint was created
    assert os.path.exists(CHECKPOINT_PATH), "Checkpoint file was not created."
    print("Training complete and checkpoint verified.")

    # 6. Inference / Prediction
    print("Generating predictions...")

    # Use a small subset of test data for the demo
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")

    # We pass the path to the engine function, but we want to ensure it runs quickly.
    # The generate_predictions function in engine.py has a debug_limit parameter.

    generate_predictions(
        model=model,
        device=device,
        test_csv_path=test_csv_path,
        output_csv_path=SUBMISSION_PATH,
        batch_size=BATCH_SIZE,
        num_workers=2,
        debug_limit=20,  # Process only first 20 samples
    )

    # 7. Submission Verification
    print("Verifying submission file...")
    assert os.path.exists(SUBMISSION_PATH), "Submission file was not created."

    submission_df = pd.read_csv(SUBMISSION_PATH)

    # Check columns
    expected_cols = ["id", "predicted"]
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(submission_df.columns)}"

    # Check content format (id should be int, predicted should be string of space-separated ints)
    assert not submission_df.empty, "Submission DataFrame is empty."
    first_row = submission_df.iloc[0]
    assert isinstance(
        first_row["id"], (int, np.integer)
    ), "ID column should be integer."
    assert isinstance(first_row["predicted"], str), "Predicted column should be string."

    # Verify we have 5 predictions per image
    preds = first_row["predicted"].split(" ")
    assert len(preds) == 5, f"Expected 5 predictions per image, got {len(preds)}"

    print("Submission verification passed.")
    print("Demo script completed successfully.")


if __name__ == "__main__":
    main()
