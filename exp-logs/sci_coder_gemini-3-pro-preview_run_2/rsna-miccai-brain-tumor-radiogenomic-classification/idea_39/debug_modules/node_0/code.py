import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

# Import provided library modules
from library import config, utils, data_loader, model, trainer


def run_demo():
    print(">>> Starting Demonstration Script")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Set random seed for reproducibility
    utils.seed_everything(42)

    # Override config for speed and demonstration purposes
    print(">>> Configuring for fast demonstration run...")
    config.NUM_EPOCHS = 1  # Train for only 1 epoch
    config.DEBUG_SAMPLE_SIZE = 16  # Use only 16 samples
    config.BATCH_SIZE = 4  # Small batch size

    # Setup working directory for this demo
    config.WORKING_DIR = "./working/demo_execution"
    config.CACHE_DIR = config.WORKING_DIR
    config.MODEL_SAVE_DIR = config.WORKING_DIR
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    print(f"Working Directory: {config.WORKING_DIR}")
    print(f"Device: {config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n>>> Loading Datasets (Texture & Context Experts)...")

    # We load datasets with load_cache=False to demonstrate the processing logic
    # This generates the 12-channel volumes from DICOM files
    train_ds_A, train_ds_B, val_ds_A, val_ds_B = data_loader.get_datasets(
        load_cache=False, debug_size=config.DEBUG_SAMPLE_SIZE
    )

    # Verify Dataset sizes
    print(f"Train Dataset A Size: {len(train_ds_A)}")
    print(f"Val Dataset A Size: {len(val_ds_A)}")

    assert len(train_ds_A) == config.DEBUG_SAMPLE_SIZE, "Train DS size mismatch"
    assert len(val_ds_A) == config.DEBUG_SAMPLE_SIZE, "Val DS size mismatch"

    # Verify Data Shapes
    # Get one sample: (Tensor(C, H, W), Tensor(Label))
    sample_vol, sample_label = train_ds_A[0]

    print(f"Sample Volume Shape: {sample_vol.shape}")
    print(f"Sample Label: {sample_label}")

    # Expected shape: (12, 224, 224)
    expected_shape = (config.NUM_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)
    assert (
        sample_vol.shape == expected_shape
    ), f"Expected {expected_shape}, got {sample_vol.shape}"
    assert isinstance(sample_vol, torch.Tensor), "Output should be a torch Tensor"

    # -------------------------------------------------------------------------
    # 3. Model Instantiation
    # -------------------------------------------------------------------------
    print("\n>>> Instantiating AsymmetricEfficientNet...")

    net = model.AsymmetricEfficientNet()
    net.to(config.DEVICE)

    # Verify Forward Pass with Dummy Data
    dummy_input = torch.randn(
        2, config.NUM_CHANNELS, config.IMG_SIZE, config.IMG_SIZE
    ).to(config.DEVICE)
    with torch.no_grad():
        output = net(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    # Expected output: (Batch_Size, 1) -> (2, 1)
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Starting Training Phase (Texture Expert - Stride 2)...")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds_A,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Use 0 workers for simple debug script to avoid multiprocessing overhead
    )
    val_loader = DataLoader(
        val_ds_A, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Initialize Trainer
    demo_trainer = trainer.Trainer(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.DEVICE,
        save_name="best_model_demo",
    )

    # Run Training
    best_model_path = demo_trainer.fit(epochs=config.NUM_EPOCHS)

    print(f"Training complete. Best model saved to: {best_model_path}")
    assert os.path.exists(best_model_path), "Best model file was not created."

    # -------------------------------------------------------------------------
    # 5. Inference & TTA Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Starting Inference Phase with TTA...")

    # For demonstration, we process a small subset of test data manually
    # to avoid processing the entire test set which takes time.
    test_data, test_ids = data_loader.process_dataset_split(
        config.TEST_METADATA,
        config.STRIDE_TEXTURE,
        "test_demo",
        load_cache=False,
        debug_size=10,  # Limit to 10 test samples
    )

    # Create Test Dataset and Loader
    test_ds = data_loader.BraTSDataset(test_data, test_ids, is_train=False)
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Load the trained model weights
    net.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))

    # Run Prediction with Test-Time Augmentation
    predictions = trainer.predict_with_tta(net, test_loader, config.DEVICE)

    print(f"Predictions generated: {len(predictions)}")
    print(f"Sample predictions: {predictions[:5]}")

    # Verify Predictions
    assert len(predictions) == 10, "Number of predictions does not match input size"
    assert np.all(
        (predictions >= 0.0) & (predictions <= 1.0)
    ), "Predictions out of probability range [0, 1]"

    # -------------------------------------------------------------------------
    # 6. Submission File Generation
    # -------------------------------------------------------------------------
    print("\n>>> Generating Sample Submission...")

    submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": predictions})

    submission_path = os.path.join(config.WORKING_DIR, "submission_demo.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to: {submission_path}")
    assert os.path.exists(submission_path), "Submission file not created"

    # Check file content
    saved_df = pd.read_csv(submission_path)
    assert saved_df.shape == (10, 2), "Submission file shape incorrect"
    assert (
        "BraTS21ID" in saved_df.columns and "MGMT_value" in saved_df.columns
    ), "Submission columns incorrect"

    print("\n>>> Demonstration Completed Successfully.")


if __name__ == "__main__":
    run_demo()
