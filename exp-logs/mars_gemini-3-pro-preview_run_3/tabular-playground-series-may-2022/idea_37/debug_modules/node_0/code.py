import sys
import os
import torch
import numpy as np
import pandas as pd
import shutil

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.data_processing import preprocess_data, get_dataloaders
from library.model import DAR_PE_Model
from library.training import train_model, generate_submission


def main():
    # ==========================================
    # 1. Configuration Setup for Demo
    # ==========================================
    print("Setting up configuration for demonstration...")

    # Override Config parameters for speed and isolation
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 128
    Config.CACHE_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up previous demo runs if they exist
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)

    # Initialize directories and random seeds via Config.setup()
    # This uses the modified paths we just set.
    Config.setup()

    # ==========================================
    # 2. Data Processing (Debug Mode)
    # ==========================================
    print("Running data processing (debug mode)...")

    # load_cached_data=False ensures we run the processing logic
    # debug=True loads a small subset (5000 train, 1000 val/test)
    data = preprocess_data(load_cached_data=False, debug=True)

    # Verification: Check data dictionary structure
    required_keys = ["X_train_cont", "X_train_cat", "y_train", "dims", "ids"]
    for key in required_keys:
        if key not in data:
            raise AssertionError(f"Missing key in processed data: {key}")

    dims = data["dims"]
    print(f"Data dimensions verified. n_cont: {dims['n_cont']}, n_cat: {dims['n_cat']}")

    # ==========================================
    # 3. Data Loaders
    # ==========================================
    print("Creating dataloaders...")
    # Use 0 workers to avoid multiprocessing overhead during this short demo
    loaders = get_dataloaders(data, batch_size=Config.BATCH_SIZE, num_workers=0)

    # Verification: Check batch shapes from the training loader
    x_cont_batch, x_cat_batch, y_batch = next(iter(loaders["train"]))

    # Assert batch size matches config
    if x_cont_batch.shape[0] != Config.BATCH_SIZE:
        # Note: It might be smaller if dataset size < batch size, but with 5000 rows and bs=128 it should be full
        if x_cont_batch.shape[0] > Config.BATCH_SIZE:
            raise AssertionError(
                f"Batch size mismatch. Expected {Config.BATCH_SIZE}, got {x_cont_batch.shape[0]}"
            )

    # Assert feature dimensions match metadata
    if x_cont_batch.shape[1] != dims["n_cont"]:
        raise AssertionError(
            f"Continuous feature dim mismatch. Expected {dims['n_cont']}, got {x_cont_batch.shape[1]}"
        )
    if x_cat_batch.shape[1] != dims["n_cat"]:
        raise AssertionError(
            f"Categorical feature dim mismatch. Expected {dims['n_cat']}, got {x_cat_batch.shape[1]}"
        )

    # ==========================================
    # 4. Model Initialization & Logic Check
    # ==========================================
    print("Initializing model and verifying forward pass...")
    model = DAR_PE_Model(n_cont=dims["n_cont"], vocab_sizes=dims["vocab_sizes"])
    model.to(Config.DEVICE)
    model.eval()

    with torch.no_grad():
        # Move batch to device
        x_cont_dev = x_cont_batch.to(Config.DEVICE)
        x_cat_dev = x_cat_batch.to(Config.DEVICE)

        # Run forward pass
        outputs = model(x_cont_dev, x_cat_dev)

        # Verify output structure (DAR-PE has 5 independent streams)
        if not isinstance(outputs, list):
            raise AssertionError(
                "Model output should be a list (one tensor per stream)."
            )

        if len(outputs) != 5:
            raise AssertionError(f"Expected 5 stream outputs, got {len(outputs)}")

        for i, out in enumerate(outputs):
            if out.shape[0] != x_cont_batch.shape[0] or out.shape[1] != 1:
                raise AssertionError(f"Stream {i} output shape mismatch: {out.shape}")

    print("Model logic verified.")

    # ==========================================
    # 5. Training Loop
    # ==========================================
    print("Starting training loop...")
    # train_model handles the loop, validation, and saving the best model
    best_model_path = train_model(loaders, dims)

    if not os.path.exists(best_model_path):
        raise AssertionError(f"Best model file was not saved at {best_model_path}")

    print(f"Training complete. Model saved to {best_model_path}")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("Generating submission...")
    test_ids = data["ids"]
    generate_submission(best_model_path, loaders, dims, test_ids)

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError("Submission file was not created.")

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    if sub_df.shape != (len(test_ids), 2):
        raise AssertionError(
            f"Submission shape mismatch. Expected ({len(test_ids)}, 2), got {sub_df.shape}"
        )

    if list(sub_df.columns) != ["id", "target"]:
        raise AssertionError(f"Invalid columns in submission: {sub_df.columns}")

    print("Submission generated and verified.")
    print("Demonstration complete.")


if __name__ == "__main__":
    main()
