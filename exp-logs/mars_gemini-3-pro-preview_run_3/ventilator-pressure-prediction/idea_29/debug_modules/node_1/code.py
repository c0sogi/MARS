import sys
import os
import torch
import numpy as np
import pandas as pd
import shutil
import warnings

# Add current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.features import FeatureEngineer
from library.dataset import get_dataloaders
from library.model import PCDRHNet
from library.loss import MaskedL1Loss
from library.train import Trainer


def main():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    print("=== Ventilator Pressure Prediction Demo ===\n")

    # 1. Configuration Overrides for Speed & Demonstration
    # We modify the Config class attributes directly to adapt to the demo constraints.
    print("1. Configuring environment...")

    # Enable Debug mode to load only a small subset of breaths
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Process only 200 breaths for speed

    # Training Hyperparameters for quick execution
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Disable multiprocessing for tiny datasets

    # Redirect outputs to a dedicated demo directory in ./working
    Config.CACHE_DIR = "./working/demo_cache/"
    Config.SUBMISSION_DIR = "./working/demo_submission/"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up previous demo runs to ensure fresh execution
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set reproducibility
    seed_everything(Config.SEED)
    device = get_device()
    print(f"   Device: {device}")
    print(f"   Debug Mode: {Config.DEBUG}")
    print(f"   Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # 2. Feature Engineering & Data Processing
    print("\n2. Running Feature Engineering...")
    fe = FeatureEngineer()

    # Process data (load_cached_data=False forces re-computation for this demo)
    data = fe.process_data(load_cached_data=False)

    # Verify Data Shapes
    # Expected: (N_Breaths, 80_Steps, N_Features)
    train_x = data["train"]["x"]
    train_y = data["train"]["y"]
    train_mask = data["train"]["mask"]

    print(f"   Train X shape: {train_x.shape}")
    print(f"   Train Y shape: {train_y.shape}")

    # Logic Verification: Check dimensions
    assert train_x.ndim == 3, "Train X should be 3D tensor (Batch, Seq, Feat)"
    assert train_x.shape[1] == 80, "Sequence length must be 80 time steps"
    assert train_x.shape[2] == len(
        Config.STREAM_A_COLS
    ), "Feature count mismatch for Stream A"
    assert train_y.shape == (train_x.shape[0], 80, 1), "Target shape mismatch"
    assert train_mask.shape == (train_x.shape[0], 80, 1), "Mask shape mismatch"
    print("   -> Data shapes verified successfully.")

    # 3. DataLoader Initialization
    print("\n3. Initializing DataLoaders...")
    # We reload using get_dataloaders to test the full pipeline integration
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,  # Load the cache we just created
    )

    # Fetch one batch to verify
    batch_x, batch_mask, batch_y = next(iter(train_loader))
    print(f"   Batch X shape: {batch_x.shape}")

    assert batch_x.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert batch_x.shape[1] == 80, "Sequence length mismatch in loader"
    print("   -> DataLoaders verified successfully.")

    # 4. Model Architecture & Forward Pass
    print("\n4. Initializing Model (PCDRH-Net)...")
    model = PCDRHNet().to(device)

    # Test Forward Pass
    dummy_input = batch_x.to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"   Model Output shape: {output.shape}")
    assert output.shape == (Config.BATCH_SIZE, 80, 1), "Model output shape incorrect"
    print("   -> Forward pass verified successfully.")

    # 5. Loss Function Verification (MaskedL1Loss)
    print("\n5. Verifying Masked Loss Logic...")
    criterion = MaskedL1Loss()

    # Create a deterministic test case
    # Prediction: 10, Target: 0 -> Absolute Error: 10
    # Case A: u_out = 0 (Inspiratory) -> Should count towards loss
    # Case B: u_out = 1 (Expiratory)  -> Should be masked (ignored)
    t_preds = torch.tensor([10.0, 10.0]).view(1, 2, 1)
    t_targets = torch.tensor([0.0, 0.0]).view(1, 2, 1)
    t_u_out = torch.tensor([0.0, 1.0]).view(1, 2, 1)  # [Keep, Ignore]

    loss = criterion(t_preds, t_targets, t_u_out)

    # Expected Calculation:
    # Error_A = |10 - 0| * 1 = 10
    # Error_B = |10 - 0| * 0 = 0
    # Mean = (10 + 0) / (1 valid sample) = 10
    print(f"   Calculated Loss: {loss.item()}")

    assert (
        abs(loss.item() - 10.0) < 1e-5
    ), f"Loss logic failed. Expected 10.0, got {loss.item()}"
    print("   -> Loss function logic verified successfully.")

    # 6. Training Pipeline
    print(f"\n6. Starting Training Loop ({Config.EPOCHS} epochs)...")
    trainer = Trainer(model, device)

    # Run training
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # Verify checkpoint creation
    assert os.path.exists(
        trainer.best_model_path
    ), "Best model checkpoint was not created."
    print("   -> Training complete and model saved.")

    # 7. Inference & Submission
    print("\n7. Generating Predictions...")
    predictions = trainer.predict(test_loader)

    print(f"   Predictions count: {len(predictions)}")
    print(f"   Test IDs count: {len(test_ids)}")

    assert len(predictions) == len(
        test_ids
    ), "Mismatch between predictions and test IDs"

    # Generate Submission DataFrame
    submission_df = pd.DataFrame(
        {Config.ID_COL: test_ids, Config.TARGET_COL: predictions}
    )

    # Save to disk
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"   Submission saved to: {Config.SUBMISSION_PATH}")

    # Verify file content
    saved_df = pd.read_csv(Config.SUBMISSION_PATH)
    print("   Submission Head:")
    print(saved_df.head(3))

    assert saved_df.shape == (len(test_ids), 2), "Submission file shape incorrect"
    assert Config.TARGET_COL in saved_df.columns, "Target column missing in submission"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
