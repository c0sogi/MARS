import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Import library components
from library.config import Config, set_seed
from library.data_processing import load_and_process_data, NFLContactDataset
from library.model import KETRNet
from library.train import Trainer
from library.utils import compute_mcc


def main():
    # 1. Setup and Configuration
    # --------------------------
    warnings.filterwarnings("ignore")

    # Set seed for reproducibility
    set_seed(42)

    print("Initializing Configuration...")

    # Define a working directory for this demo
    demo_artifact_dir = "./working/demo_execution"

    # Clean up previous demo runs if they exist
    if os.path.exists(demo_artifact_dir):
        shutil.rmtree(demo_artifact_dir)
    os.makedirs(demo_artifact_dir)

    # Patch the Config class defaults at runtime.
    # This ensures that when library functions instantiate Config() internally,
    # they use our optimized settings for speed and location.
    Config.artifact_dir = demo_artifact_dir
    Config.debug = True
    Config.debug_sample_size = 200  # Small subset for rapid execution
    Config.epochs = 2  # Minimal epochs for demonstration
    Config.batch_size = 16  # Small batch size
    Config.patience = 1  # Aggressive early stopping for demo

    # 2. Data Processing Demonstration
    # --------------------------------
    print("\n=== Testing Data Processing ===")

    # Load and process data (internally uses the patched Config)
    # We force load_cached_data=False to demonstrate the processing logic
    train_dataset, train_meta = load_and_process_data(
        split="train", debug=True, load_cached_data=False
    )
    val_dataset, val_meta = load_and_process_data(
        split="validation", debug=True, load_cached_data=False
    )

    # Verification
    assert isinstance(
        train_dataset, NFLContactDataset
    ), "Train dataset is not correct type"
    assert (
        len(train_dataset) == Config.debug_sample_size
    ), f"Expected {Config.debug_sample_size} samples, got {len(train_dataset)}"
    assert (
        len(val_dataset) == Config.debug_sample_size
    ), f"Expected {Config.debug_sample_size} samples, got {len(val_dataset)}"

    # Verify Data Shapes
    # Expected: (Batch, Window_Size=11, Features=13) based on default config
    sample_x, sample_y = train_dataset[0]
    expected_window = 11
    expected_features = 13

    assert sample_x.shape == (
        expected_window,
        expected_features,
    ), f"Feature shape mismatch. Expected ({expected_window}, {expected_features}), got {sample_x.shape}"
    assert isinstance(sample_y, torch.Tensor), "Label is not a tensor"
    assert sample_y.numel() == 1, "Label should be a scalar"

    print(
        f"Data loaded successfully. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}"
    )

    # 3. Model Architecture Demonstration
    # -----------------------------------
    print("\n=== Testing Model Architecture ===")

    # Instantiate model using a config instance (inherits our patched defaults)
    model_config = Config()
    model = KETRNet(model_config)

    # Create dummy input for verification
    # Shape: (Batch, Window, Features)
    dummy_batch_size = 4
    dummy_input = torch.randn(
        dummy_batch_size, model_config.window_size, len(model_config.feature_cols)
    )

    # Forward pass
    output = model(dummy_input)

    # Verification
    assert output.shape == (
        dummy_batch_size,
    ), f"Model output shape mismatch. Expected ({dummy_batch_size},), got {output.shape}"
    assert output.requires_grad, "Model output detached from graph unexpectedly"

    print("Model forward pass successful.")

    # 4. Training Loop Demonstration
    # ------------------------------
    print("\n=== Testing Training Loop ===")

    # Initialize Trainer
    trainer = Trainer(model_config)

    # Execute training
    # This runs the training loop, validation, and threshold optimization
    best_threshold = trainer.fit(train_dataset, val_dataset)

    # Verification
    assert isinstance(
        best_threshold, (float, np.floating)
    ), "Threshold should be a float"
    assert 0.0 < best_threshold < 1.0, f"Threshold out of bounds: {best_threshold}"

    # Check for artifacts
    model_path = os.path.join(demo_artifact_dir, "best_model.pth")
    threshold_path = os.path.join(demo_artifact_dir, "best_threshold.npy")

    assert os.path.exists(model_path), "Model checkpoint was not saved"
    assert os.path.exists(threshold_path), "Threshold file was not saved"

    print(f"Training complete. Best threshold: {best_threshold:.4f}")

    # 5. Inference Demonstration
    # --------------------------
    print("\n=== Testing Inference ===")

    # Load the best model state
    model.load_state_dict(torch.load(model_path, map_location=trainer.device))
    model.to(trainer.device)
    model.eval()

    # Run inference on validation set manually to verify
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=Config.batch_size)
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(trainer.device)
            logits = model(X)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    # Apply threshold
    preds = (all_probs >= best_threshold).astype(int)

    # Calculate metric
    mcc = compute_mcc(all_targets, preds)

    print(f"Inference successful. Validation MCC on subset: {mcc:.4f}")
    print("\nAll demonstrations passed successfully.")


if __name__ == "__main__":
    main()
