import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data_loader import load_data, IcebergDataset
from library.model import SC_WBN, predict
from library.train import run_fold


def demo_pipeline():
    # 1. Configuration Overrides for Speed
    print("--- Setting up Demo Configuration ---")
    seed_everything(42)

    # Override Config parameters to run a fast demo
    Config.DEBUG = True
    Config.MAX_DEBUG_SAMPLES = 32  # Small subset for quick execution
    Config.NUM_EPOCHS = 1  # Single epoch
    Config.BATCH_SIZE = 4  # Small batch size

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    print("\n--- Loading Data ---")
    # load_data handles caching and debug slicing internally based on Config
    data = load_data(load_cached_data=True)

    (X_train, y_train, inc_train, X_val, y_val, inc_val, X_test, inc_test, ids_test) = (
        data
    )

    # 3. Verify Data Integrity
    print("\n--- Verifying Data Shapes ---")
    print(f"Train Images: {X_train.shape}")
    print(f"Train Labels: {y_train.shape}")

    # Assertions to ensure debug slicing worked and shapes are correct
    assert (
        len(X_train) == Config.MAX_DEBUG_SAMPLES
    ), "Train set size mismatch for debug mode"
    assert X_train.shape[1:] == (
        3,
        75,
        75,
    ), f"Unexpected image dimensions: {X_train.shape[1:]}"
    assert len(y_train) == Config.MAX_DEBUG_SAMPLES, "Label size mismatch"
    assert len(X_test) == Config.MAX_DEBUG_SAMPLES, "Test set size mismatch"

    # Verify Dataset Class
    ds = IcebergDataset(X_train, inc_train, y_train, transform=True)
    sample_img, sample_inc, sample_lbl = ds[0]
    assert sample_img.shape == (3, 75, 75), "Dataset image tensor shape mismatch"
    assert isinstance(sample_inc, torch.Tensor), "Incidence angle should be a tensor"
    print("Data validation passed.")

    # 4. Model Initialization & Forward Pass
    print("\n--- Initializing Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SC_WBN().to(device)

    # Dummy forward pass to check architecture
    dummy_img = torch.randn(4, 3, 75, 75).to(device)
    dummy_inc = torch.randn(4).to(device)

    with torch.no_grad():
        output = model(dummy_img, dummy_inc)

    assert output.shape == (
        4,
        1,
    ), f"Model output shape mismatch. Expected (4, 1), got {output.shape}"
    print("Model forward pass successful.")

    # 5. Training Demonstration
    print("\n--- Running Training Loop (1 Epoch) ---")
    # run_fold encapsulates the training logic for one fold
    best_loss = run_fold(
        fold_idx=0,
        X_train=X_train,
        inc_train=inc_train,
        y_train=y_train,
        X_val=X_val,
        inc_val=inc_val,
        y_val=y_val,
        device=device,
    )
    print(f"Training finished. Best Validation Loss: {best_loss:.4f}")

    # 6. Prediction
    print("\n--- Generating Predictions ---")
    # Load the model saved by run_fold
    model_path = os.path.join(Config.WORKING_DIR, "sc_wbn_fold_0.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))

    # Prepare Test Loader
    test_dataset = IcebergDataset(X_test, inc_test, transform=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Generate predictions
    preds = predict(model, test_loader, device)

    # Verify predictions
    assert len(preds) == len(ids_test), "Prediction count does not match test set size"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions contain values outside [0, 1]"

    print(f"Generated {len(preds)} predictions.")
    print(f"Sample predictions: {preds[:5]}")

    # 7. Save Submission
    print("\n--- Saving Submission ---")
    submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": preds})

    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    demo_pipeline()
