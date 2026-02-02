import os
import shutil
import pandas as pd
import numpy as np
import torch
from library.config import Config
from library.utils import seed_everything, MaskedL1Loss
from library.dataset import prepare_data
from library.model import BidirectionalLSTM
from library.trainer import Trainer


def main():
    print("=== Ventilator Pressure Prediction Pipeline Demo ===")

    # 1. Configuration Setup for Demo
    # We modify the Config global state to optimize for a quick demonstration run.
    print(">>> Configuring environment...")
    Config.DEBUG = True
    Config.DEBUG_BREATHS = 50  # Use only 50 breaths per split for speed
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 16
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # We will create a specific sample submission file for this debug run
    Config.SAMPLE_SUBMISSION_PATH = os.path.join(
        Config.WORKING_DIR, "sample_submission.csv"
    )

    # Clean up working directory if it exists to ensure a fresh run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Create necessary directories
    Config.setup()

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    # 2. Data Preparation
    print("\n>>> Preparing Data (Debug Mode)...")
    # This loads data, scales it, reshapes it, and returns DataLoaders
    train_loader, val_loader, test_loader = prepare_data(
        load_cached_data=True, batch_size=Config.BATCH_SIZE
    )

    # Verification: Check DataLoaders are not empty
    assert len(train_loader) > 0, "Train loader should not be empty."
    assert len(val_loader) > 0, "Validation loader should not be empty."
    assert len(test_loader) > 0, "Test loader should not be empty."

    # Verification: Check Batch Shapes
    # Fetch one batch from training loader
    batch = next(iter(train_loader))
    x, y, u_out = batch["x"], batch["y"], batch["u_out"]

    print(f"  Input batch shape: {x.shape}")
    print(f"  Target batch shape: {y.shape}")

    # Expected shape: (Batch, Seq_Len, Features) -> (16, 80, 5)
    assert x.shape == (
        Config.BATCH_SIZE,
        Config.SEQUENCE_LENGTH,
        Config.INPUT_DIM,
    ), f"Input shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQUENCE_LENGTH, Config.INPUT_DIM)}, got {x.shape}"
    assert y.shape == (
        Config.BATCH_SIZE,
        Config.SEQUENCE_LENGTH,
    ), f"Target shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQUENCE_LENGTH)}, got {y.shape}"
    assert u_out.shape == (
        Config.BATCH_SIZE,
        Config.SEQUENCE_LENGTH,
    ), f"u_out shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQUENCE_LENGTH)}, got {u_out.shape}"

    # 3. Create Dummy Sample Submission
    # The predict_and_submit function expects a sample submission file that matches the length of predictions.
    # Since we are using a subset (DEBUG mode), the provided sample_submission.csv is too large.
    # We generate a dummy one matching the test_loader size.
    total_test_samples = len(test_loader.dataset) * Config.SEQUENCE_LENGTH
    print(
        f"\n>>> Creating dummy sample submission for {total_test_samples} time steps..."
    )

    dummy_submission = pd.DataFrame(
        {
            "id": np.arange(1, total_test_samples + 1),
            "pressure": np.zeros(total_test_samples),
        }
    )
    dummy_submission.to_csv(Config.SAMPLE_SUBMISSION_PATH, index=False)

    # 4. Model Initialization & Verification
    print("\n>>> Initializing Model...")
    model = BidirectionalLSTM(
        input_dim=Config.INPUT_DIM,
        hidden_dim=64,  # Reduced size for demo
        num_layers=1,
        bidirectional=True,
    )

    # Move to configured device
    device = Config.DEVICE
    model.to(device)

    # Manual Forward Pass Check
    print("  Verifying forward pass...")
    model.eval()
    with torch.no_grad():
        # Move input batch to device
        x_device = x.to(device)
        output = model(x_device)

    print(f"  Model output shape: {output.shape}")
    assert output.shape == (
        Config.BATCH_SIZE,
        Config.SEQUENCE_LENGTH,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQUENCE_LENGTH)}, got {output.shape}"

    # 5. Loss Function Verification
    print("\n>>> Verifying Masked L1 Loss Logic...")
    criterion = MaskedL1Loss()

    # Test Case:
    # Time step 0: u_out=0 (Inspiratory), Pred=10, Target=20 -> Error=10
    # Time step 1: u_out=1 (Expiratory),  Pred=100, Target=20 -> Error=80 (Should be masked/ignored)
    # Expected Loss = 10 / 1 = 10.0

    t_pred = torch.tensor([[10.0, 100.0]], device=device)
    t_target = torch.tensor([[20.0, 20.0]], device=device)
    t_u_out = torch.tensor([[0.0, 1.0]], device=device)

    loss_val = criterion(t_pred, t_target, t_u_out)
    print(f"  Calculated Loss: {loss_val.item()}")

    assert torch.isclose(
        loss_val, torch.tensor(10.0, device=device)
    ), f"Loss calculation incorrect. Expected 10.0, got {loss_val.item()}"
    print("  Loss function verified.")

    # 6. Training Loop
    print("\n>>> Starting Training (Trainer Loop)...")
    trainer = Trainer(model, Config)

    # Run training for 1 epoch
    trainer.fit(train_loader, val_loader)

    # Check if model file was saved (implies validation ran)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"  Model saved successfully at {Config.MODEL_SAVE_PATH}")
    else:
        print(
            "  Model was not saved (Validation loss might not have improved in 1 epoch with random weights)."
        )

    # 7. Inference & Submission
    print("\n>>> Generating Predictions...")
    trainer.predict(test_loader)

    # Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"  Submission file created at {Config.SUBMISSION_PATH}")
        print(f"  Submission shape: {sub_df.shape}")

        assert (
            len(sub_df) == total_test_samples
        ), f"Submission length mismatch. Expected {total_test_samples}, got {len(sub_df)}"

        # Check for NaNs
        assert not sub_df.isnull().values.any(), "Submission contains NaN values."
        print("  Submission file verified.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
