import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import (
    set_seed,
    TRAIN_META_PATH,
    SEQ_LEN,
    N_FEATURES,
    DEVICE,
    CACHE_DIR,
)
from library.utils import angles_to_vector, vector_to_angles, angular_dist_score
from library.dataset import IceCubeDataset
from library.model import NeutrinoBiGRU
from library.loss import CosineDistanceLoss
from library.engine import train_with_early_stopping


def main():
    # 1. Setup and Reproducibility
    print("Setting seed for reproducibility...")
    set_seed(42)

    # =========================================================================
    # 2. Verify Utility Functions
    # =========================================================================
    print("\n[1/5] Verifying Utility Functions...")

    # Test 1: Coordinate Conversion (Round Trip)
    # Define some known angles: Azimuth=pi/2 (y-axis), Zenith=pi/2 (xy-plane) -> (0, 1, 0)
    az_true = np.array([np.pi / 2])
    zen_true = np.array([np.pi / 2])

    vec = angles_to_vector(az_true, zen_true)
    print(f"  Angles (pi/2, pi/2) -> Vector: {vec[0]}")

    # Expected vector is roughly [0, 1, 0]
    assert np.allclose(vec, [[0, 1, 0]], atol=1e-6), "Vector conversion failed"

    # Convert back
    az_rec, zen_rec = vector_to_angles(vec[:, 0], vec[:, 1], vec[:, 2])
    print(f"  Vector {vec[0]} -> Angles: ({az_rec[0]:.4f}, {zen_rec[0]:.4f})")

    assert np.allclose(az_rec, az_true, atol=1e-6), "Azimuth reconstruction failed"
    assert np.allclose(zen_rec, zen_true, atol=1e-6), "Zenith reconstruction failed"

    # Test 2: Angular Distance Score
    # Distance between identical angles should be 0
    score = angular_dist_score(
        np.column_stack([az_true, zen_true]), np.column_stack([az_rec, zen_rec])
    )
    print(f"  Angular Distance (Identical): {score:.6f}")
    assert score < 1e-6, "Angular distance for identical vectors should be ~0"

    # Distance between orthogonal vectors (e.g., x-axis vs y-axis) should be pi/2
    vec_x = np.array([[1, 0, 0]])
    az_x, zen_x = vector_to_angles(vec_x[:, 0], vec_x[:, 1], vec_x[:, 2])

    score_ortho = angular_dist_score(
        np.column_stack([az_true, zen_true]), np.column_stack([az_x, zen_x])
    )
    print(
        f"  Angular Distance (Orthogonal): {score_ortho:.4f} (Expected: {np.pi/2:.4f})"
    )
    assert np.isclose(score_ortho, np.pi / 2, atol=1e-4), "Angular distance logic error"

    # =========================================================================
    # 3. Verify Dataset Loading
    # =========================================================================
    print("\n[2/5] Verifying Dataset Loading...")

    # Load metadata
    print(f"  Loading metadata from {TRAIN_META_PATH}...")
    meta_df = pd.read_parquet(TRAIN_META_PATH)

    # Filter for a small subset to ensure speed (Batch 1)
    # We select 200 events from batch_id 1
    subset_df = meta_df[meta_df["batch_id"] == 1].head(200).copy()
    print(f"  Created subset with {len(subset_df)} events from Batch 1.")

    # Instantiate Dataset
    dataset = IceCubeDataset(subset_df, mode="train", cache_limit=1)

    # Fetch one sample
    features, targets = dataset[0]

    print(f"  Feature Shape: {features.shape} (Expected: [{SEQ_LEN}, {N_FEATURES}])")
    print(f"  Target Shape: {targets.shape} (Expected: [2])")

    # Assertions
    assert features.shape == (SEQ_LEN, N_FEATURES), "Incorrect feature tensor shape"
    assert targets.shape == (2,), "Incorrect target tensor shape"
    assert isinstance(features, torch.Tensor), "Features should be a torch Tensor"
    assert isinstance(targets, torch.Tensor), "Targets should be a torch Tensor"

    # Check for NaNs
    assert not torch.isnan(features).any(), "Features contain NaNs"
    assert not torch.isnan(targets).any(), "Targets contain NaNs"

    # =========================================================================
    # 4. Verify Model and Loss
    # =========================================================================
    print("\n[3/5] Verifying Model and Loss...")

    model = NeutrinoBiGRU().to(DEVICE)
    criterion = CosineDistanceLoss()

    # Create a dummy batch
    batch_size = 4
    dummy_input = (
        features.unsqueeze(0).repeat(batch_size, 1, 1).to(DEVICE)
    )  # (4, 128, 6)
    dummy_target = targets.unsqueeze(0).repeat(batch_size, 1).to(DEVICE)  # (4, 2)

    # Forward Pass
    pred_vector = model(dummy_input)
    print(f"  Model Output Shape: {pred_vector.shape} (Expected: [{batch_size}, 3])")

    assert pred_vector.shape == (batch_size, 3), "Model output shape mismatch"

    # Loss Calculation
    loss = criterion(pred_vector, dummy_target)
    print(f"  Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    # =========================================================================
    # 5. Verify Training Engine
    # =========================================================================
    print("\n[4/5] Verifying Training Engine...")

    # Split subset into train/val
    train_size = int(0.8 * len(subset_df))
    train_df = subset_df.iloc[:train_size]
    val_df = subset_df.iloc[train_size:]

    train_ds = IceCubeDataset(train_df, mode="train")
    val_ds = IceCubeDataset(val_df, mode="train")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Run training for 2 epochs
    print("  Starting short training run (2 epochs)...")
    model, history = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=2,
        patience=1,
        device=DEVICE,
        save_path=os.path.join(CACHE_DIR, "demo_model.pth"),
    )

    print("  Training history keys:", history.keys())
    assert len(history["train_loss"]) > 0, "Training history is empty"
    assert len(history["val_loss"]) > 0, "Validation history is empty"

    # =========================================================================
    # 6. Verify Inference
    # =========================================================================
    print("\n[5/5] Verifying Inference Logic...")

    model.eval()
    with torch.no_grad():
        # Use the dummy input from before
        preds = model(dummy_input)

        # Normalize
        preds_norm = torch.nn.functional.normalize(preds, p=2, dim=1).cpu().numpy()

        # Convert to angles
        az_pred, zen_pred = vector_to_angles(
            preds_norm[:, 0], preds_norm[:, 1], preds_norm[:, 2]
        )

        print("  Inference Predictions (First 2 samples):")
        for i in range(2):
            print(f"    Sample {i}: Azimuth={az_pred[i]:.4f}, Zenith={zen_pred[i]:.4f}")

        assert az_pred.shape == (batch_size,), "Predicted azimuth shape mismatch"
        assert zen_pred.shape == (batch_size,), "Predicted zenith shape mismatch"

    print("\nAll verifications passed successfully!")


if __name__ == "__main__":
    main()
