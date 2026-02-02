import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders, process_f27
from library.model import HybridSwiGLUNet
from library.train import Trainer


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Set seed for reproducibility
    seed_everything(42)

    # Override Config parameters to ensure the demo runs quickly (< 5 mins)
    # We use a separate working directory for the demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update paths in Config to point to the demo directory
    Config.PROCESSED_DATA_PATH = os.path.join(Config.WORKING_DIR, "processed_data.npz")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Limit data size and training duration
    Config.DEBUG_SAMPLE_SIZE = 500  # Use only 500 samples
    Config.BATCH_SIZE = 16  # Small batch size
    Config.EPOCHS = 2  # Only 2 epochs
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo execution

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # --------------------------------------------------------------------------
    # 2. Logic Verification: Feature Engineering
    # --------------------------------------------------------------------------
    print("\n[2] Verifying Feature Engineering Logic...")

    # Test process_f27 function (String -> Integer Indices)
    dummy_input = pd.Series(["ABCDEFGHIJ", "ZZZZZZZZZZ"])
    processed_output = process_f27(dummy_input)

    # Validation
    # Shape should be (2, 10) as sequence length is 10
    assert processed_output.shape == (
        2,
        10,
    ), f"Unexpected shape: {processed_output.shape}"
    # 'A' (ord 65) -> 1, 'B' (ord 66) -> 2
    assert processed_output[0, 0] == 1, "Mapping for 'A' is incorrect"
    assert processed_output[0, 1] == 2, "Mapping for 'B' is incorrect"
    # 'Z' (ord 90) -> 26
    assert processed_output[1, 0] == 26, "Mapping for 'Z' is incorrect"

    print("Feature engineering logic (process_f27) verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Data Loading
    # --------------------------------------------------------------------------
    print("\n[3] Initializing DataLoaders...")

    # We force load_cached_data=False to demonstrate processing from raw CSVs
    # The DEBUG_SAMPLE_SIZE in Config will automatically slice the data after processing
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify DataLoader output
    batch = next(iter(train_loader))
    cont_x = batch["cont_features"]
    cat_x = batch["cat_features"]
    targets = batch["target"]

    print(
        f"Batch Shapes -> Continuous: {cont_x.shape}, Categorical: {cat_x.shape}, Targets: {targets.shape}"
    )

    # Assertions for shapes
    # Continuous features: 30 (f_00 to f_30 excluding f_27)
    assert cont_x.shape == (Config.BATCH_SIZE, 30), "Continuous feature shape mismatch"
    # Categorical sequence length: 10
    assert cat_x.shape == (Config.BATCH_SIZE, 10), "Categorical feature shape mismatch"
    # Targets
    assert targets.shape[0] == Config.BATCH_SIZE, "Target batch size mismatch"

    print("DataLoaders initialized and verified.")

    # --------------------------------------------------------------------------
    # 4. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = HybridSwiGLUNet().to(device)
    model.eval()

    # Create dummy inputs on the correct device
    dummy_cont = torch.randn(Config.BATCH_SIZE, 30).to(device)
    dummy_cat = torch.randint(1, 27, (Config.BATCH_SIZE, 10)).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_cont, dummy_cat)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
    assert (
        output.min() >= 0.0 and output.max() <= 1.0
    ), "Model output not in [0, 1] range (Sigmoid check)"

    print("Model architecture forward pass verified.")

    # --------------------------------------------------------------------------
    # 5. Training Loop Simulation
    # --------------------------------------------------------------------------
    print("\n[5] Executing Training Loop...")

    trainer = Trainer(device)

    # Run training (fit)
    # This uses the parameters set in Config (2 Epochs)
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # Verify model artifact creation
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH} after training."
        )

    print(f"Training complete. Model saved to {Config.MODEL_PATH}")

    # --------------------------------------------------------------------------
    # 6. Inference and Submission
    # --------------------------------------------------------------------------
    print("\n[6] Running Inference...")

    # Predict on test set
    submission_df = trainer.predict(test_loader)

    print("Sample Submission:")
    print(submission_df.head())

    # Verify Submission Format
    assert "id" in submission_df.columns, "Submission missing 'id' column"
    assert "target" in submission_df.columns, "Submission missing 'target' column"
    assert len(submission_df) == len(
        test_loader.dataset
    ), "Submission row count mismatch"

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
    else:
        raise FileNotFoundError("Failed to save submission file.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
