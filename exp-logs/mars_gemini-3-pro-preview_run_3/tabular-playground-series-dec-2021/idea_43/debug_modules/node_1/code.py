import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_device
from library.model import DualViewDCNResNet
from library.data_loader import get_dataloaders
from library.train import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_mini_dataset(source_path, dest_path, n_samples=1000):
    """
    Helper to create a small subset of the data for demonstration purposes.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    df = pd.read_parquet(source_path)
    # Sample without replacement
    df_mini = df.sample(n=min(n_samples, len(df)), random_state=42).reset_index(
        drop=True
    )
    df_mini.to_parquet(dest_path, index=False)
    print(f"Created mini dataset at {dest_path} with {len(df_mini)} rows.")
    return len(df_mini)


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Setting up Configuration for Demo...")

    # Define a specific directory for this demo execution to avoid cache conflicts
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config parameters for speed
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir
    Config.BEST_MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 128
    Config.NUM_WORKERS = (
        0  # Use 0 workers for simple debugging/demo to avoid multiprocessing overhead
    )
    Config.SCHEDULER_PATIENCE = 1

    # Set seeds
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Prepare Mini Datasets
    # -------------------------------------------------------------------------
    print("\n[2] Preparing Mini Datasets...")

    # Define paths for mini datasets
    mini_train_path = os.path.join(demo_dir, "train.parquet")
    mini_val_path = os.path.join(demo_dir, "val.parquet")
    mini_test_path = os.path.join(demo_dir, "test.parquet")

    # Create subsets from the actual metadata
    # We use the paths defined in the original Config to find the source
    create_mini_dataset(
        os.path.join("./metadata", "train.parquet"), mini_train_path, n_samples=2000
    )
    create_mini_dataset(
        os.path.join("./metadata", "val.parquet"), mini_val_path, n_samples=500
    )
    create_mini_dataset(
        os.path.join("./metadata", "test.parquet"), mini_test_path, n_samples=500
    )

    # Point Config to these new mini files
    Config.TRAIN_PATH = mini_train_path
    Config.VAL_PATH = mini_val_path
    Config.TEST_PATH = mini_test_path

    # -------------------------------------------------------------------------
    # 3. Data Loading Pipeline
    # -------------------------------------------------------------------------
    print("\n[3] Running Data Loading Pipeline...")

    # load_cached_data=False forces the pipeline to process our new mini datasets
    train_loader, val_loader, test_loader, input_dim = get_dataloaders(
        load_cached_data=False
    )

    # Verification
    print(f"Input Dimension: {input_dim}")
    assert input_dim > 0, "Input dimension should be positive."

    # Check a single batch
    X_batch, y_batch = next(iter(train_loader))
    print(f"Train Batch X shape: {X_batch.shape}")
    print(f"Train Batch y shape: {y_batch.shape}")

    assert (
        X_batch.shape[0] == Config.BATCH_SIZE
    ), f"Batch size mismatch. Expected {Config.BATCH_SIZE}, got {X_batch.shape[0]}"
    assert X_batch.shape[1] == input_dim, "Feature dimension mismatch."
    assert isinstance(X_batch, torch.Tensor), "X should be a Tensor."
    assert isinstance(y_batch, torch.Tensor), "y should be a Tensor."

    # -------------------------------------------------------------------------
    # 4. Model Instantiation & Verification
    # -------------------------------------------------------------------------
    print("\n[4] Initializing Model...")

    model = DualViewDCNResNet(input_dim=input_dim, num_classes=Config.NUM_CLASSES)
    model.to(device)

    # Dummy Forward Pass
    dummy_input = torch.randn(4, input_dim).to(device)
    logits, aux_logits = model(dummy_input)

    print(f"Logits shape: {logits.shape}")
    if aux_logits is not None:
        print(f"Aux Logits shape: {aux_logits.shape}")

    assert logits.shape == (4, Config.NUM_CLASSES), "Logits output shape incorrect."
    # Aux logits might be None depending on architecture flow, but if present should match
    if aux_logits is not None:
        assert aux_logits.shape == (
            4,
            Config.NUM_CLASSES,
        ), "Aux logits output shape incorrect."

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Starting Training Demo...")

    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.fit()

    # Verify model checkpoint creation
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not saved."
    print("Training finished and model saved successfully.")

    # -------------------------------------------------------------------------
    # 6. Inference & Submission Generation
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference on Test Set...")

    # Load the best model
    best_model = DualViewDCNResNet(input_dim=input_dim, num_classes=Config.NUM_CLASSES)
    state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
    best_model.load_state_dict(state_dict)
    best_model.to(device)
    best_model.eval()

    predictions = []
    ids_list = []

    with torch.no_grad():
        for batch_X, batch_ids in test_loader:
            batch_X = batch_X.to(device)
            # Forward pass
            logits, _ = best_model(batch_X)
            # Get predictions (argmax)
            preds = torch.argmax(logits, dim=1)

            # Map back to 1-7 range (model predicts 0-6)
            preds = preds + 1

            predictions.extend(preds.cpu().numpy())
            ids_list.extend(batch_ids.numpy())

    # Create submission DataFrame
    df_sub = pd.DataFrame({"Id": ids_list, "Cover_Type": predictions})

    # Save submission
    submission_path = os.path.join(demo_dir, "submission_demo.csv")
    df_sub.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print(df_sub.head())

    # Verify submission file
    assert os.path.exists(submission_path), "Submission file not created."
    loaded_sub = pd.read_csv(submission_path)
    assert len(loaded_sub) == len(ids_list), "Submission length mismatch."
    assert (
        "Id" in loaded_sub.columns and "Cover_Type" in loaded_sub.columns
    ), "Submission columns mismatch."
    assert (
        loaded_sub["Cover_Type"].min() >= 1 and loaded_sub["Cover_Type"].max() <= 7
    ), "Invalid class labels in submission."

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
