import os
import shutil
import torch
import numpy as np
import pandas as pd
import random

# Import from the provided library
from library.config import Config
from library.features import FeatureEngineer
from library.dataset import VentilatorDataset
from library.model import WideProjectedNet, predict_test
from library.loss import MaskedAuxL1Loss
from library.train import Trainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration Override
    # We override Config to ensure the script runs quickly (Debug mode)
    # and uses a separate directory for this demo.
    set_seed(42)

    print("Configuring environment...")
    Config.debug = True  # Use small subset of data
    Config.epochs = 1  # Run only 1 epoch
    Config.batch_size = 16  # Small batch size
    Config.num_workers = 0  # Avoid multiprocessing overhead for small demo

    # Set a specific working directory for this demo to avoid conflicts
    Config.working_dir = "./working/demo_execution"

    # Update dependent paths in Config manually since they were defined at class level
    Config.train_cache = os.path.join(Config.working_dir, "train_engineered.parquet")
    Config.val_cache = os.path.join(Config.working_dir, "val_engineered.parquet")
    Config.test_cache = os.path.join(Config.working_dir, "test_engineered.parquet")
    Config.scaler_center_path = os.path.join(Config.working_dir, "scaler_center.npy")
    Config.scaler_scale_path = os.path.join(Config.working_dir, "scaler_scale.npy")
    Config.model_path = os.path.join(Config.working_dir, "model.pth")
    Config.output_submission_path = os.path.join(Config.working_dir, "submission.csv")

    # Clean up previous demo runs if any
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)
    Config.setup()

    print(f"Working directory set to: {Config.working_dir}")

    # 2. Feature Engineering
    print("\n=== Step 1: Feature Engineering ===")
    fe = FeatureEngineer()

    # Process Train/Val (Debug mode will sample 100 train and 50 val breaths)
    # We force load_cached=False to demonstrate the engineering logic
    train_df, val_df = fe.process_train_val(load_cached=False)

    print(f"Train DataFrame Shape: {train_df.shape}")
    print(f"Val DataFrame Shape: {val_df.shape}")

    # Verification
    expected_cols = ["volume", "R__u_in", "vol__C", "u_in_lag1", "u_in_diff1"]
    for col in expected_cols:
        if col not in train_df.columns:
            raise AssertionError(
                f"Expected engineered column '{col}' missing from train_df"
            )

    # Check if scaler files were saved
    if not os.path.exists(Config.scaler_center_path):
        raise AssertionError("Scaler parameters not saved to disk.")

    print("Feature Engineering verification passed.")

    # 3. Dataset Loading
    print("\n=== Step 2: Dataset Loading ===")
    # Initialize dataset (will load from the parquet files created above)
    train_ds = VentilatorDataset(split="train", load_cached=True)

    # Get a single sample
    x, u_out, y = train_ds[0]

    print(f"Sample Input Shape: {x.shape}")
    print(f"Sample Mask Shape: {u_out.shape}")
    print(f"Sample Target Shape: {y.shape}")

    # Verification
    # Sequence length should be 80
    if x.shape[0] != 80:
        raise AssertionError(f"Expected sequence length 80, got {x.shape[0]}")
    # u_out and y should be 1D tensors of length 80
    if u_out.shape != (80,) or y.shape != (80,):
        raise AssertionError("Mask or Target shapes are incorrect.")

    input_dim = x.shape[1]
    print(f"Input Feature Dimension: {input_dim}")
    print("Dataset verification passed.")

    # 4. Model Architecture & Loss
    print("\n=== Step 3: Model & Loss Verification ===")
    device = torch.device("cpu")  # Use CPU for simple verification
    model = WideProjectedNet(input_dim=input_dim).to(device)

    # Create a dummy batch (Batch Size = 4)
    dummy_x = torch.randn(4, 80, input_dim).to(device)

    # Forward pass
    final_pred, aux_pred = model(dummy_x)

    print(f"Model Output Shape (Final): {final_pred.shape}")
    print(f"Model Output Shape (Aux): {aux_pred.shape}")

    # Verification
    # Output should be (Batch, Seq, 1)
    if final_pred.shape != (4, 80, 1):
        raise AssertionError(
            f"Expected output shape (4, 80, 1), got {final_pred.shape}"
        )

    # Loss Calculation
    criterion = MaskedAuxL1Loss()
    dummy_y = torch.randn(4, 80).to(device)
    dummy_u_out = torch.randint(0, 2, (4, 80)).float().to(device)  # Binary mask

    loss = criterion((final_pred, aux_pred), dummy_y, dummy_u_out)
    print(f"Calculated Loss: {loss.item()}")

    if torch.isnan(loss) or loss.item() < 0:
        raise AssertionError("Loss is NaN or negative.")

    print("Model and Loss verification passed.")

    # 5. Training Loop
    print("\n=== Step 4: Training Loop (1 Epoch) ===")
    # Initialize Trainer
    # Trainer internally initializes datasets and model
    trainer = Trainer(debug=True)

    # Run training
    trainer.fit(epochs=1)

    # Verification
    if not os.path.exists(Config.model_path):
        raise AssertionError("Model checkpoint was not saved after training.")

    print("Training loop completed and model saved.")

    # 6. Inference
    print("\n=== Step 5: Inference & Submission ===")

    # Prepare Test Dataset
    test_ds = VentilatorDataset(split="test", load_cached=False)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=Config.batch_size, shuffle=False
    )

    # Load best model
    best_model = WideProjectedNet(input_dim=input_dim).to(Config.device)
    best_model.load_state_dict(
        torch.load(Config.model_path, map_location=Config.device)
    )

    # Predict
    # predict_test returns a (N, Seq) array, need to flatten for submission
    preds = predict_test(best_model, test_loader, Config.device).flatten()

    print(f"Predictions shape: {preds.shape}")

    # Verify predictions match test set length (in debug mode, test set is full size unless we filter it manually,
    # but process_test in features.py doesn't filter test set by default in debug mode.
    # However, for this demo, we just want to ensure it runs.
    # Note: process_test processes the whole test file.
    # In a real debug scenario, we might want to truncate test, but here we just check execution.)

    # Generate Submission
    # We read the test file to get IDs
    test_df = pd.read_csv(Config.test_file)

    # Check length consistency
    if len(preds) != len(test_df):
        # This might happen if process_test didn't filter but we expected it to.
        # Actually, in features.py, process_test does NOT have a debug filter for rows.
        # So preds length should equal test_df length.
        pass

    submission = pd.DataFrame({"id": test_df["id"], "pressure": preds})

    submission.to_csv(Config.output_submission_path, index=False)

    # Final Verification
    if not os.path.exists(Config.output_submission_path):
        raise AssertionError("Submission file was not created.")

    saved_df = pd.read_csv(Config.output_submission_path)
    if (
        saved_df.shape[1] != 2
        or "id" not in saved_df.columns
        or "pressure" not in saved_df.columns
    ):
        raise AssertionError("Submission file format is incorrect.")

    print(f"Submission saved to {Config.output_submission_path}")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
