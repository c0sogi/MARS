import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model_arch as model_arch
import library.trainer as trainer
import library.inference as inference


def run_demo():
    print("Starting MNSHDNetwork Demo...")

    # 1. Setup and Configuration Overrides
    # Set seed for reproducibility
    utils.seed_everything(config.SEED)

    # Force DEBUG_SAMPLE_SIZE to a small number to ensure the demo runs quickly.
    # We modify the variable in both config and data_loader to ensure it takes effect
    # regardless of how it was imported.
    DEMO_SAMPLE_SIZE = 6
    config.DEBUG_SAMPLE_SIZE = DEMO_SAMPLE_SIZE
    data_loader.DEBUG_SAMPLE_SIZE = DEMO_SAMPLE_SIZE

    # Define a demo working directory to avoid conflicts with existing runs
    DEMO_WORKING_DIR = os.path.join(config.WORKING_DIR, "demo_run")
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    # Override paths in config/modules to point to demo directory
    config.WORKING_DIR = DEMO_WORKING_DIR
    data_loader.WORKING_DIR = DEMO_WORKING_DIR
    trainer.WORKING_DIR = DEMO_WORKING_DIR
    inference.WORKING_DIR = DEMO_WORKING_DIR

    config.MODEL_SAVE_PATH = os.path.join(DEMO_WORKING_DIR, "best_model.pth")
    trainer.MODEL_SAVE_PATH = config.MODEL_SAVE_PATH
    inference.MODEL_SAVE_PATH = config.MODEL_SAVE_PATH

    config.SUBMISSION_FILE_PATH = os.path.join(DEMO_WORKING_DIR, "submission.csv")
    inference.SUBMISSION_FILE_PATH = config.SUBMISSION_FILE_PATH

    print(f"Debug Mode: Sample Size set to {DEMO_SAMPLE_SIZE}")
    print(f"Working Directory: {DEMO_WORKING_DIR}")

    # 2. Data Loading Demonstration
    print("\n[Step 1] Loading Datasets...")
    # get_datasets handles caching. We force reload to ensure we use the debug sample size.
    # Note: In a real run, load_cached_data=True is preferred for speed.
    train_dataset, val_dataset, test_dataset = data_loader.get_datasets(
        load_cached_data=False
    )

    # Verify Dataset sizes
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    # Assertions to verify data loading logic
    assert (
        len(train_dataset) <= DEMO_SAMPLE_SIZE
    ), "Train dataset size exceeds debug limit"
    assert len(val_dataset) <= DEMO_SAMPLE_SIZE, "Val dataset size exceeds debug limit"

    # Verify Data Shapes
    # Fetch one sample
    if len(train_dataset) > 0:
        sample_img, sample_label = train_dataset[0]
        print(f"Sample Input Shape: {sample_img.shape}")
        print(f"Sample Label: {sample_label}")

        # Expected shape: (128, 224, 224) -> (Channels, H, W)
        assert sample_img.shape == (
            128,
            224,
            224,
        ), f"Incorrect input shape: {sample_img.shape}"
        assert isinstance(sample_label, torch.Tensor), "Label should be a tensor"

    # 3. Model Architecture Demonstration
    print("\n[Step 2] Initializing Model...")
    device = config.DEVICE
    model = model_arch.MNSHDNetwork().to(device)

    # Verify Model Output Shape with Dummy Data
    batch_size = 2
    dummy_input = torch.randn(batch_size, 128, 224, 224).to(device)

    print("Running forward pass verification...")
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        batch_size,
        1,
    ), f"Expected output shape (B, 1), got {output.shape}"

    # 4. Training Loop Demonstration
    print("\n[Step 3] Running Training Loop (1 Epoch)...")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # Setup Optimizer and Loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Run one epoch of training
    # We use the function provided in library.trainer
    train_loss = trainer.train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    print(f"Train Loss: {train_loss:.4f}")

    # Run validation
    val_loss, val_auc = trainer.validate(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

    # Save the model (required for inference step)
    print(f"Saving model to {config.MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    assert os.path.exists(config.MODEL_SAVE_PATH), "Model file was not saved."

    # 5. Inference Demonstration
    print("\n[Step 4] Running Inference...")

    # We need to ensure the test dataset is available for the inference module.
    # The inference module calls get_datasets internally.
    # Since we set data_loader.DEBUG_SAMPLE_SIZE, it will use that.
    # We pass load_cached_data=True so it picks up the cache we just generated in Step 1
    # (though technically Step 1 generated cache for train/val/test).

    try:
        inference.generate_submission(
            model_path=config.MODEL_SAVE_PATH,
            output_path=config.SUBMISSION_FILE_PATH,
            load_cached_data=True,
        )
    except Exception as e:
        print(f"Inference failed with error: {e}")
        raise e

    # Verify Submission File
    if os.path.exists(config.SUBMISSION_FILE_PATH):
        df_sub = pd.read_csv(config.SUBMISSION_FILE_PATH)
        print(f"Submission file created at {config.SUBMISSION_FILE_PATH}")
        print(df_sub.head())

        expected_cols = ["BraTS21ID", "MGMT_value"]
        assert (
            list(df_sub.columns) == expected_cols
        ), f"Invalid columns: {df_sub.columns}"
        assert len(df_sub) > 0, "Submission file is empty"
        print("Submission verification passed.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run_demo()
