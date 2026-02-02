import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.data_utils import preprocess_pipeline, TabularDataset, set_seed
from library.model import SelfNormalizingFunnelNet
from library.train_eval import train_model, generate_submission


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("Setting up configuration for demonstration...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 2000  # Small subset for quick execution
    Config.EPOCHS = 1  # Single epoch for demo
    Config.BATCH_SIZE = 128

    # Redirect paths to a demo-specific working directory
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = (
        Config.WORKING_DIR
    )  # Save submission in working dir for demo

    # We must manually update dependent paths since they were defined at class level
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_processed.parquet")
    Config.PREPROCESSOR_CACHE = os.path.join(Config.WORKING_DIR, "metadata.npy")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Create directories
    Config.setup()

    # Set reproducible seed
    set_seed(Config.SEED)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # ==========================================
    # 2. Data Preprocessing
    # ==========================================
    print("\nRunning preprocessing pipeline...")
    # Force re-computation to demonstrate logic (load_cached_data=False)
    train_df, val_df, test_df, vocab_sizes, cat_cols, cont_cols = preprocess_pipeline(
        load_cached_data=False
    )

    # Validation: Check shapes
    print("Validating processed data shapes...")
    # In debug mode, we expect DEBUG_SAMPLES rows
    assert (
        len(train_df) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} training samples, got {len(train_df)}"
    assert (
        len(val_df) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} validation samples, got {len(val_df)}"
    assert (
        len(test_df) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} test samples, got {len(test_df)}"

    # Validation: Check feature columns
    assert len(vocab_sizes) == len(
        cat_cols
    ), "Mismatch between vocab_sizes and categorical columns."
    # f_27 decomposed (10) + f_29 + f_30 = 12 categorical columns
    assert (
        len(cat_cols) == 12
    ), f"Expected 12 categorical columns, found {len(cat_cols)}"

    print("Data preprocessing validation passed.")

    # ==========================================
    # 3. Dataset & DataLoader
    # ==========================================
    print("\nInitializing Datasets and DataLoaders...")

    train_dataset = TabularDataset(train_df, cat_cols, cont_cols, target_col="target")
    val_dataset = TabularDataset(val_df, cat_cols, cont_cols, target_col="target")
    test_dataset = TabularDataset(test_df, cat_cols, cont_cols, target_col=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for simple demo stability
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Validation: Check batch structure
    print("Validating batch structure...")
    sample_x_cat, sample_x_cont, sample_y = next(iter(train_loader))

    assert sample_x_cat.shape == (
        Config.BATCH_SIZE,
        len(cat_cols),
    ), "Incorrect categorical input shape"
    assert sample_x_cont.shape == (
        Config.BATCH_SIZE,
        len(cont_cols),
    ), "Incorrect continuous input shape"
    assert sample_y.shape == (Config.BATCH_SIZE,), "Incorrect target shape"

    print("Batch structure validation passed.")

    # ==========================================
    # 4. Model Initialization
    # ==========================================
    print("\nInitializing Model...")

    model = SelfNormalizingFunnelNet(vocab_sizes=vocab_sizes, cont_dim=len(cont_cols))
    model.to(Config.DEVICE)

    # Validation: Forward pass
    print("Validating model forward pass...")
    with torch.no_grad():
        # Move sample to device
        sample_x_cat = sample_x_cat.to(Config.DEVICE)
        sample_x_cont = sample_x_cont.to(Config.DEVICE)

        logits = model(sample_x_cat, sample_x_cont)

        assert logits.shape == (
            Config.BATCH_SIZE,
            1,
        ), f"Expected output shape ({Config.BATCH_SIZE}, 1), got {logits.shape}"

    print("Model initialization validation passed.")

    # ==========================================
    # 5. Training Loop
    # ==========================================
    print("\nStarting Training Loop (Demo)...")

    # train_model handles the loop, validation, and saving best model
    trained_model = train_model(model, train_loader, val_loader)

    # Validation: Check if model file was created
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model file was not saved after training."
    print(f"Training complete. Model saved to {Config.MODEL_SAVE_PATH}")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("\nGenerating Submission...")

    # We need to ensure the sample submission file exists or handle the warning in generate_submission
    # The provided environment has sample_submission.csv in ./input

    generate_submission(trained_model, test_loader)

    # Validation: Check submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    # Since we used DEBUG mode, the test_loader only has DEBUG_SAMPLES predictions.
    # However, generate_submission loads the *original* sample_submission.csv (100k rows)
    # and assigns predictions.
    # If len(preds) != len(submission_df), the library code prints a warning but might crash or misalign
    # if not handled strictly.
    # In this specific library implementation:
    # submission_df["target"] = probs
    # This assignment will fail if lengths differ in pandas.
    #
    # NOTE: The provided library code `generate_submission` does:
    # submission_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    # submission_df["target"] = probs
    #
    # Since we are running in DEBUG mode, `probs` will have length 2000.
    # `submission_df` (from input) has length 100000.
    # This assignment will raise a ValueError.
    #
    # To make this demo run successfully without crashing on that pandas assignment,
    # we must mock the `Config.SAMPLE_SUBMISSION_PATH` to point to a file that matches our debug size,
    # OR we accept that `generate_submission` is designed for full runs.
    #
    # Fix for Demo: Create a dummy sample submission matching the debug test set size.

    print("Demo: Creating dummy sample submission to match debug test size...")
    dummy_sample_sub = pd.DataFrame(
        {"id": test_df["id"].values, "target": [0.5] * len(test_df)}
    )
    dummy_sub_path = os.path.join(Config.WORKING_DIR, "sample_submission_debug.csv")
    dummy_sample_sub.to_csv(dummy_sub_path, index=False)

    # Temporarily point Config to this dummy file so generate_submission works
    Config.SAMPLE_SUBMISSION_PATH = dummy_sub_path

    # Retry submission generation with correct sizes
    generate_submission(trained_model, test_loader)

    # Re-verify
    final_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(final_sub) == Config.DEBUG_SAMPLES, "Submission length mismatch."

    print(f"Submission generation successful. Saved to {Config.SUBMISSION_PATH}")
    print("\n=== Demo Execution Complete Successfully ===")


if __name__ == "__main__":
    run_demo()
