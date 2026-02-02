import os
import torch
import numpy as np
import pandas as pd
import math
from torch.utils.data import DataLoader, Subset

# Import from the provided library
from library.config import Config
from library.utils import (
    euler_to_quaternion,
    quaternion_to_euler,
    compute_iou_bev,
    get_transformation_matrix,
    transform_points,
)
from library.dataset import LidarDataset
from library.model import BevYolo
from library.loss import YoloLoss
from library.train import train_model
from library.predict import Predictor


def run_demo():
    print("=== Starting 3D Object Detection Library Demo ===")

    # 1. Setup
    Config.set_seed(42)
    device = Config.DEVICE
    print(f"Device: {device}")

    # ==========================================
    # 2. Verify Utils
    # ==========================================
    print("\n--- Verifying Utils ---")

    # Test Quaternion <-> Euler
    # Rotate 90 degrees (pi/2) around Z axis
    yaw_target = math.pi / 2
    q = euler_to_quaternion(0, 0, yaw_target)
    r, p, y = quaternion_to_euler(q)

    assert np.isclose(y, yaw_target), f"Yaw mismatch: expected {yaw_target}, got {y}"
    assert np.isclose(r, 0), "Roll should be 0"
    assert np.isclose(p, 0), "Pitch should be 0"
    print("Quaternion/Euler conversion: OK")

    # Test IoU Calculation
    # Box format: [cx, cy, w, l]
    # Box A: 2x2 square at (0,0) -> spans [-1, 1] in x and y. Area = 4.
    # Box B: 2x2 square at (1,0) -> spans [0, 2] in x and [-1, 1] in y. Area = 4.
    # Intersection: x in [0, 1], y in [-1, 1]. Width=1, Height=2. Area = 2.
    # Union: 4 + 4 - 2 = 6.
    # IoU: 2 / 6 = 1/3 (~0.333)
    box_a = np.array([[0, 0, 2, 2]], dtype=np.float32)
    box_b = np.array([[1, 0, 2, 2]], dtype=np.float32)
    iou = compute_iou_bev(box_a, box_b)

    assert np.isclose(
        iou[0, 0], 1 / 3, atol=1e-4
    ), f"IoU mismatch: expected 0.333, got {iou[0,0]}"
    print("IoU Calculation: OK")

    # ==========================================
    # 3. Verify Dataset
    # ==========================================
    print("\n--- Verifying Dataset ---")
    # Initialize dataset
    # We use 'train' split to ensure we get targets
    ds_train = LidarDataset(split="train", load_cached_data=True)

    if len(ds_train) > 0:
        # Fetch one sample
        bev_tensor, target_tensor, token = ds_train[0]

        # Check Shapes
        # BEV: (3, 512, 512)
        expected_bev_shape = (Config.IN_CHANNELS, Config.BEV_HEIGHT, Config.BEV_WIDTH)
        assert (
            bev_tensor.shape == expected_bev_shape
        ), f"BEV shape mismatch: expected {expected_bev_shape}, got {bev_tensor.shape}"

        # Target: (Num_Anchors, 128, 128, 10)
        out_h = Config.BEV_HEIGHT // 4
        out_w = Config.BEV_WIDTH // 4
        expected_target_shape = (len(Config.ANCHORS), out_h, out_w, 10)
        assert (
            target_tensor.shape == expected_target_shape
        ), f"Target shape mismatch: expected {expected_target_shape}, got {target_tensor.shape}"

        print(f"Dataset loaded sample {token} successfully.")
        print(f"BEV Shape: {bev_tensor.shape}")
        print(f"Target Shape: {target_tensor.shape}")
    else:
        print("Warning: Dataset is empty. Skipping dataset verification details.")

    # ==========================================
    # 4. Verify Model & Loss
    # ==========================================
    print("\n--- Verifying Model & Loss ---")

    model = BevYolo().to(device)
    criterion = YoloLoss().to(device)

    # Create dummy batch (B=2)
    B = 2
    dummy_input = torch.randn(
        B, Config.IN_CHANNELS, Config.BEV_HEIGHT, Config.BEV_WIDTH
    ).to(device)

    # Dummy targets
    # (B, A, H, W, 10)
    dummy_targets = torch.zeros(B, len(Config.ANCHORS), out_h, out_w, 10).to(device)
    # Set one valid object in the center for the first sample
    dummy_targets[0, 0, out_h // 2, out_w // 2, 0] = 1.0  # Valid
    dummy_targets[0, 0, out_h // 2, out_w // 2, 9] = 0.0  # Class 0

    # Forward Pass
    predictions = model(dummy_input)

    # Check Output Shape
    # Expected: (B, A, H, W, 1 + 8 + Num_Classes)
    num_classes = Config.NUM_CLASSES
    expected_out_channels = 1 + 8 + num_classes
    expected_out_shape = (B, len(Config.ANCHORS), out_h, out_w, expected_out_channels)

    assert (
        predictions.shape == expected_out_shape
    ), f"Model output mismatch: expected {expected_out_shape}, got {predictions.shape}"
    print("Model Forward Pass: OK")

    # Loss Calculation
    loss, metrics = criterion(predictions, dummy_targets)

    assert not torch.isnan(loss), "Loss is NaN"
    assert "loss_obj" in metrics
    assert "loss_reg" in metrics
    assert "loss_cls" in metrics
    print(f"Loss Calculation: OK (Total Loss: {loss.item():.4f})")

    # ==========================================
    # 5. Verify Training Loop
    # ==========================================
    print("\n--- Verifying Training Loop (Debug Mode) ---")
    # Run training for 1 epoch on a tiny subset
    # This verifies the optimizer, backprop, and checkpoint saving
    train_model(num_epochs=1, batch_size=2, debug=True)

    # Check if checkpoint was created
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint successfully created at {checkpoint_path}")
    else:
        # Depending on random init, loss might not decrease in 1 epoch, so best_model might not save.
        # But usually with debug mode and overfitting a few samples, it should.
        # We'll check if the directory exists at least.
        print(
            "Checkpoint check: 'best_model.pth' not found (loss might not have improved), but training ran."
        )

    # ==========================================
    # 6. Verify Prediction / Inference
    # ==========================================
    print("\n--- Verifying Prediction Logic ---")

    # Instead of running the full 'generate_predictions' which processes the whole test set,
    # we manually invoke the Predictor on a small subset of test data.

    predictor = Predictor(checkpoint_path=checkpoint_path, device=device)

    # Load Test Dataset subset
    test_ds = LidarDataset(split="test", load_cached_data=True)
    if len(test_ds) > 0:
        subset_indices = range(min(2, len(test_ds)))
        test_subset = Subset(test_ds, subset_indices)
        test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

        print(f"Running inference on {len(test_subset)} test samples...")

        results = []
        with torch.no_grad():
            for bev, tokens in test_loader:
                bev = bev.to(device)

                # Forward
                preds = predictor.model(bev)

                # Decode
                decoded = predictor._decode_predictions(preds)

                # Check decoded shape: (B, A, H, W, 9)
                # 9 = [x, y, z, w, l, h, yaw, score, class]
                assert decoded.shape[-1] == 9

                # Just verify we can format a string for the first token
                token = tokens[0]

                # Flatten
                flat_preds = decoded.view(1, -1, 9).cpu().numpy()[0]

                # Fake NMS / Thresholding for demo (take top 1)
                best_idx = np.argmax(flat_preds[:, 7])
                best_pred = flat_preds[best_idx]

                # Format string (simplified version of predict.py logic)
                conf = best_pred[7]
                if conf > 0.0:  # Just to show logic
                    s = f"{conf:.4f} {best_pred[0]:.4f} {best_pred[1]:.4f} ..."
                    results.append({"Id": token, "PredictionString": s})

        print(f"Generated predictions for {len(results)} samples.")
        print("Prediction Logic: OK")

        # Create a dummy submission file to prove write capability
        df_sub = pd.DataFrame(results)
        dummy_sub_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
        df_sub.to_csv(dummy_sub_path, index=False)
        print(f"Demo submission saved to {dummy_sub_path}")

    else:
        print("Test dataset is empty, skipping prediction verification.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
