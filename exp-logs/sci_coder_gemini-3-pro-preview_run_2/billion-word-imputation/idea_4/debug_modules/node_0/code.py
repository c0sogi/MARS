import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset

# Import library components
from library.config import Config
from library.utils import set_seed, setup_logger, save_submission
from library.data_factory import create_dataloaders
from library.modeling import GapLocatorModel, InFillerModel
from library.engine import train_locator, train_infiller
from library.pipeline import Predictor


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print(">>> [1/6] Configuring Environment for Demo Run...")

    # Override Config for a fast, self-contained demo
    Config.DEBUG = True
    Config.DEBUG_SIZE = 500  # Small subset for quick execution
    Config.LOCATOR_EPOCHS = 1
    Config.INFILLER_EPOCHS = 1
    Config.LOCATOR_BATCH_SIZE = 8
    Config.INFILLER_BATCH_SIZE = 8

    # Update paths to isolate this run
    Config.PROJECT_NAME = "demo_run"
    Config.WORKING_DIR = f"./working/{Config.PROJECT_NAME}/"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Manually create directories (simulating __post_init__)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Setup logger
    logger = setup_logger("demo", os.path.join(Config.WORKING_DIR, "demo.log"))
    logger.info("Configuration updated for demo mode.")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print(">>> [2/6] Loading Data (Debug Mode)...")

    # Force reload to ensure we create new debug caches
    (
        train_loc_loader,
        val_loc_loader,
        train_fill_loader,
        val_fill_loader,
        test_loader,
    ) = create_dataloaders(load_cached_data=False)

    # Validation: Check DataLoaders
    print("Verifying DataLoader shapes...")
    batch_loc = next(iter(train_loc_loader))
    assert "input_ids" in batch_loc
    assert "labels" in batch_loc
    assert batch_loc["input_ids"].shape[0] <= Config.LOCATOR_BATCH_SIZE
    print(f"  Locator Batch: {batch_loc['input_ids'].shape}")

    batch_fill = next(iter(train_fill_loader))
    assert "input_ids" in batch_fill
    assert batch_fill["input_ids"].shape[0] <= Config.INFILLER_BATCH_SIZE
    print(f"  InFiller Batch: {batch_fill['input_ids'].shape}")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation & Forward Pass Check
    # -------------------------------------------------------------------------
    print(">>> [3/6] Verifying Model Architectures...")
    device = Config.DEVICE

    # Test Locator
    loc_model = GapLocatorModel(Config.LOCATOR_MODEL).to(device)
    loc_out = loc_model(
        batch_loc["input_ids"].to(device),
        batch_loc["attention_mask"].to(device),
        batch_loc["labels"].to(device),
    )
    assert "loss" in loc_out
    assert loc_out["loss"] is not None
    print(f"  Locator Forward Pass Loss: {loc_out['loss'].item():.4f}")

    # Test InFiller
    fill_model = InFillerModel(Config.INFILLER_MODEL).to(device)
    fill_out = fill_model(
        batch_fill["input_ids"].to(device),
        batch_fill["attention_mask"].to(device),
        batch_fill["labels"].to(device),
    )
    assert hasattr(fill_out, "loss")
    print(f"  InFiller Forward Pass Loss: {fill_out.loss.item():.4f}")

    # Cleanup to save memory
    del loc_model, fill_model, loc_out, fill_out
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print(">>> [4/6] Running Training Loops...")

    # Train Locator
    print("  Training Locator...")
    loc_save_path = train_locator(
        train_loc_loader,
        val_loc_loader,
        epochs=Config.LOCATOR_EPOCHS,
        lr=1e-4,
        device=device,
    )
    assert os.path.exists(loc_save_path), "Locator checkpoint failed to save."

    # Train InFiller
    print("  Training InFiller...")
    fill_save_path = train_infiller(
        train_fill_loader,
        val_fill_loader,
        epochs=Config.INFILLER_EPOCHS,
        lr=1e-4,
        device=device,
    )
    assert os.path.exists(fill_save_path), "InFiller checkpoint failed to save."

    # -------------------------------------------------------------------------
    # 5. Inference Pipeline
    # -------------------------------------------------------------------------
    print(">>> [5/6] Running Inference Pipeline...")

    # Initialize Predictor with trained models
    predictor = Predictor(loc_save_path, fill_save_path, device=device)

    # Create a small subset of test data for rapid inference verification
    # We take the first 10 samples from the test dataset
    test_subset_indices = range(10)
    test_subset = Subset(test_loader.dataset, test_subset_indices)
    test_subset_loader = DataLoader(
        test_subset,
        batch_size=Config.LOCATOR_BATCH_SIZE,
        shuffle=False,
        collate_fn=None,  # Use default collation as per original loader
    )

    # Run Prediction
    df_results = predictor.predict(test_subset_loader)

    # Validate Results
    assert len(df_results) == 10
    assert "id" in df_results.columns
    assert "sentence" in df_results.columns

    print("  Sample Prediction:")
    print(df_results.head(1).to_string(index=False))

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    print(">>> [6/6] Generating Submission File...")

    save_submission(
        df_results["id"].tolist(),
        df_results["sentence"].tolist(),
        Config.SUBMISSION_PATH,
    )

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Final format check
    df_check = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(df_check) == 10
    assert list(df_check.columns) == ["id", "sentence"]

    print(f"  Submission saved and verified at: {Config.SUBMISSION_PATH}")
    print(">>> Demo Completed Successfully.")


if __name__ == "__main__":
    main()
