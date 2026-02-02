import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from the provided library files
from library.utils import seed_everything, get_device
from library.dataset import load_data, IcebergDataset, get_transforms
from library.model import APCNN
from library.engine import fit_fold


def run_demo():
    # 1. Setup
    print("Initializing Demo...")
    seed_everything(42)
    device = get_device()

    # Define directories
    DEMO_DIR = "./working/demo_usage"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # 2. Data Loading (Subset for speed)
    print("\n--- Loading Data Subsets ---")
    # Load a small sample of training data (e.g., 100 samples)
    X_train, angles_train, y_train = load_data(
        "train", load_cached_data=False, sample_size=100
    )
    # Load a small sample of validation data (e.g., 50 samples)
    X_val, angles_val, y_val = load_data("val", load_cached_data=False, sample_size=50)
    # Load a small sample of test data (e.g., 20 samples)
    X_test, angles_test, ids_test = load_data(
        "test", load_cached_data=False, sample_size=20
    )

    # Verify shapes
    print(
        f"Train shapes: X={X_train.shape}, angles={angles_train.shape}, y={y_train.shape}"
    )
    print(f"Val shapes:   X={X_val.shape}, angles={angles_val.shape}, y={y_val.shape}")
    print(
        f"Test shapes:  X={X_test.shape}, angles={angles_test.shape}, ids={ids_test.shape}"
    )

    assert X_train.shape == (100, 75, 75, 3), "X_train shape mismatch"
    assert len(angles_train) == 100, "angles_train length mismatch"
    assert len(y_train) == 100, "y_train length mismatch"

    # 3. Component Verification
    print("\n--- Verifying Dataset and Model ---")

    # Test Dataset Class
    train_ds = IcebergDataset(
        X_train, angles_train, y_train, transform=get_transforms("train"), mode="train"
    )
    img, angle, target = train_ds[0]

    print(f"Dataset sample 0: Img Tensor {img.shape}, Angle {angle}, Target {target}")
    assert img.shape == (3, 75, 75), "Dataset image tensor shape incorrect"
    assert isinstance(angle, torch.Tensor), "Angle should be a tensor"
    assert isinstance(target, torch.Tensor), "Target should be a tensor"

    # Test Model Forward Pass
    model = APCNN().to(device)
    # Create a dummy batch
    dummy_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    batch_imgs, batch_angles, batch_targets = next(iter(dummy_loader))
    batch_imgs = batch_imgs.to(device)
    batch_angles = batch_angles.to(device)

    with torch.no_grad():
        output = model(batch_imgs, batch_angles)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (4, 1), "Model output shape should be (Batch_Size, 1)"

    # 4. Training Demonstration
    print("\n--- Running Training Loop (Demo) ---")
    # We use very few epochs and small batch size just to verify the pipeline runs
    best_score = fit_fold(
        fold=0,
        X_train=X_train,
        angles_train=angles_train,
        y_train=y_train,
        X_val=X_val,
        angles_val=angles_val,
        y_val=y_val,
        epochs=2,
        batch_size=16,
        patience=2,
        lr=1e-3,
        save_dir=DEMO_DIR,
    )
    print(f"Training finished. Best Score: {best_score}")

    expected_model_path = os.path.join(DEMO_DIR, "model_fold_0.pth")
    assert os.path.exists(expected_model_path), "Model checkpoint was not saved."

    # 5. Inference Demonstration
    print("\n--- Running Inference (Demo) ---")

    # Load the trained model
    model = APCNN().to(device)
    model.load_state_dict(torch.load(expected_model_path, map_location=device))
    model.eval()

    # Prepare Test Loader
    test_ds = IcebergDataset(
        X_test, angles_test, ids_test, transform=get_transforms("test"), mode="test"
    )
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    predictions = []
    test_ids_list = []

    with torch.no_grad():
        for images, angles, ids in test_loader:
            images = images.to(device)
            angles = angles.to(device)

            outputs = model(images, angles)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            predictions.extend(probs)
            test_ids_list.extend(ids)

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": test_ids_list, "is_iceberg": predictions})

    print("Inference completed. Sample predictions:")
    print(df_sub.head())

    # Save submission
    sub_path = os.path.join(DEMO_DIR, "demo_submission.csv")
    df_sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

    # 6. Final Validation
    assert len(df_sub) == 20, "Submission should have 20 rows (matching sample_size)"
    assert (
        "id" in df_sub.columns and "is_iceberg" in df_sub.columns
    ), "Submission columns missing"
    assert (
        df_sub["is_iceberg"].min() >= 0 and df_sub["is_iceberg"].max() <= 1
    ), "Probabilities out of bounds"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
