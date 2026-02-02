import os
import torch
import numpy as np
import pandas as pd
import shutil
from library.utils import seed_everything, load_data
from library.data import IcebergDataset, get_kfold_loaders, get_test_loader
from library.model import DPCNet
from library.engine import train_fold


def run_demo():
    print("----------------------------------------------------------------")
    print("Starting Iceberg Classification Library Demo")
    print("----------------------------------------------------------------")

    # 1. Setup
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define working directory for demo artifacts
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # 2. Data Loading (Debug Mode)
    print("\n[Step 1] Loading Data (Debug Mode)...")
    # debug=True loads a small subset (100 train, 50 val, 50 test)
    data_dict = load_data(load_cached_data=False, debug=True)

    # Validation of loaded data
    required_keys = [
        "X_train",
        "y_train",
        "meta_train",
        "X_test",
        "meta_test",
        "test_ids",
    ]
    for key in required_keys:
        assert key in data_dict, f"Missing key {key} in data dictionary"

    print(f"  X_train shape: {data_dict['X_train'].shape}")
    print(f"  y_train shape: {data_dict['y_train'].shape}")

    # Assertions for data integrity
    assert data_dict["X_train"].shape[1:] == (75, 75, 3), "Incorrect image dimensions"
    assert len(data_dict["X_train"]) == len(
        data_dict["y_train"]
    ), "Mismatch in X and y lengths"
    assert not np.isnan(
        data_dict["meta_train"]
    ).any(), "Found NaNs in metadata after imputation"

    # 3. Dataset and DataLoader
    print("\n[Step 2] Verifying Dataset and DataLoader...")
    # Create loaders using the provided utility
    loaders = get_kfold_loaders(data_dict, batch_size=8, n_splits=2, seed=42)
    train_loader, val_loader = loaders[0]

    print(f"  Number of folds generated: {len(loaders)}")
    print(f"  Train loader length (batches): {len(train_loader)}")

    # Fetch one batch to verify structure
    inputs, targets = next(iter(train_loader))
    imgs, angles = inputs

    print(f"  Batch Image Shape: {imgs.shape}")
    print(f"  Batch Angle Shape: {angles.shape}")
    print(f"  Batch Target Shape: {targets.shape}")

    assert imgs.shape == (
        8,
        3,
        75,
        75,
    ), "Incorrect batch image tensor shape (N, C, H, W)"
    assert angles.shape == (8, 1), "Incorrect batch angle tensor shape"
    assert targets.shape == (8, 1), "Incorrect batch target tensor shape"

    # 4. Model Architecture
    print("\n[Step 3] Verifying DPCNet Architecture...")
    model = DPCNet().to(device)

    # Perform a dummy forward pass
    dummy_img = torch.randn(4, 3, 75, 75).to(device)
    dummy_angle = torch.randn(4, 1).to(device)

    with torch.no_grad():
        output = model((dummy_img, dummy_angle))

    print(f"  Model Output Shape: {output.shape}")
    assert output.shape == (4, 1), "Model output shape mismatch, expected (Batch, 1)"

    # 5. Training Loop (Single Fold, 1 Epoch)
    print("\n[Step 4] Demonstrating Training Loop (1 Epoch)...")

    # We use the 'train_fold' function from library.engine
    # Using a small patience and 1 epoch for speed
    model_path, best_loss = train_fold(
        fold_idx=0,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=1,
        learning_rate=1e-3,
        patience=1,
        output_dir=demo_dir,
    )

    print(f"  Training completed. Best Val Loss: {best_loss:.4f}")
    print(f"  Model saved to: {model_path}")
    assert os.path.exists(model_path), "Model file was not saved correctly"

    # 6. Inference and Submission
    print("\n[Step 5] Demonstrating Inference...")

    test_loader = get_test_loader(data_dict, batch_size=8)
    test_ids = data_dict["test_ids"]

    # Load the trained model
    inference_model = DPCNet().to(device)
    inference_model.load_state_dict(torch.load(model_path))
    inference_model.eval()

    preds = []
    with torch.no_grad():
        for inputs in test_loader:
            img, angle = inputs
            img = img.to(device)
            angle = angle.to(device)

            outputs = inference_model((img, angle))
            probs = torch.sigmoid(outputs)
            preds.extend(probs.cpu().numpy().flatten())

    preds = np.array(preds)

    print(f"  Predictions generated: {len(preds)}")
    assert len(preds) == len(
        test_ids
    ), "Number of predictions does not match number of test IDs"
    assert np.all((preds >= 0) & (preds <= 1)), "Probabilities out of range [0, 1]"

    # Create submission dataframe
    sub_df = pd.DataFrame({"id": test_ids, "is_iceberg": preds})
    sub_path = os.path.join(demo_dir, "demo_submission.csv")
    sub_df.to_csv(sub_path, index=False)

    print(f"  Submission saved to: {sub_path}")
    print(f"  Submission head:\n{sub_df.head()}")

    print("\n----------------------------------------------------------------")
    print("Demo execution completed successfully.")
    print("----------------------------------------------------------------")


if __name__ == "__main__":
    run_demo()
