import os
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import DEVICE, IMG_SIZE, BATCH_SIZE, SUBMISSION_PATH, WORKING_DIR
from library.utils import seed_everything, probabilistic_f1
from library.data import get_dataloaders
from library.model import PyramidSiameseEfficientNet
from library.trainer import Trainer
from library.inference import generate_predictions

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("==== Starting Library Usage Demo ====")

    # 1. Reproducibility
    print("\n[Step 1] Setting Seeds...")
    seed_everything(42)
    print("Seeds set.")

    # 2. Data Loading (Subset)
    print("\n[Step 2] Loading DataLoaders (max_samples=16)...")
    # We use a small subset to ensure the demo finishes quickly
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, max_samples=16
    )

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches:   {len(val_loader)}")
    print(f"Test Loader Batches:  {len(test_loader)}")

    # 3. Verify Data Batch
    print("\n[Step 3] Verifying Data Batch Structure...")
    # Fetch one batch
    target_img, contra_img, labels, pred_ids = next(iter(train_loader))

    # Check shapes
    # Expected: (BATCH_SIZE, 3, 768, 768)
    expected_shape = (BATCH_SIZE, 3, IMG_SIZE[0], IMG_SIZE[1])

    print(f"Target Image Shape: {target_img.shape}")
    print(f"Contra Image Shape: {contra_img.shape}")
    print(f"Labels Shape:       {labels.shape}")

    assert (
        target_img.shape == expected_shape
    ), f"Target image shape mismatch. Expected {expected_shape}, got {target_img.shape}"
    assert (
        contra_img.shape == expected_shape
    ), f"Contralateral image shape mismatch. Expected {expected_shape}, got {contra_img.shape}"
    assert len(labels) == BATCH_SIZE, "Label batch size mismatch."

    print("Data batch structure verified.")

    # 4. Model Initialization
    print("\n[Step 4] Initializing PyramidSiameseEfficientNet...")
    model = PyramidSiameseEfficientNet()
    model = model.to(DEVICE)
    print("Model initialized and moved to device.")

    # 5. Forward Pass Verification
    print("\n[Step 5] Running Forward Pass on Batch...")
    target_img = target_img.to(DEVICE)
    contra_img = contra_img.to(DEVICE)

    with torch.no_grad():
        logits = model(target_img, contra_img)

    print(f"Logits Shape: {logits.shape}")

    # Expected output: (BATCH_SIZE, 1)
    assert logits.shape == (
        BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected {(BATCH_SIZE, 1)}, got {logits.shape}"

    print("Forward pass successful.")

    # 6. Training Loop Demo
    print("\n[Step 6] Running Training Loop (1 Epoch)...")
    trainer = Trainer(model, train_loader, val_loader, test_loader)

    # Run for 1 epoch only
    trainer.fit(epochs=1)

    # Check if best model was saved
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Training complete. Best model saved at: {best_model_path}")
    else:
        # It's possible validation loss didn't improve if initialized high,
        # but usually it saves at least once.
        print("Training complete. (No model improvement saved in this short run)")

    # 7. Inference Demo
    print("\n[Step 7] Running Inference Pipeline...")
    # This function re-initializes loaders internally, so we pass max_samples again
    df_submission = generate_predictions(load_cached_data=False, max_samples=16)

    # Verify submission file
    if os.path.exists(SUBMISSION_PATH):
        print(f"Submission file generated at: {SUBMISSION_PATH}")
        df_check = pd.read_csv(SUBMISSION_PATH)
        print("Submission Head:")
        print(df_check.head())

        # Check columns
        assert "prediction_id" in df_check.columns
        assert "cancer" in df_check.columns
        assert len(df_check) > 0
    else:
        raise FileNotFoundError("Submission file was not created.")

    # 8. Metric Verification
    print("\n[Step 8] Verifying Probabilistic F1 Metric...")
    # Case 1: Perfect prediction
    y_true = np.array([1, 0, 1, 0])
    y_pred_perfect = np.array([1.0, 0.0, 1.0, 0.0])
    score_perfect = probabilistic_f1(y_true, y_pred_perfect)
    print(f"pF1 (Perfect): {score_perfect:.4f}")
    assert np.isclose(score_perfect, 1.0), "pF1 should be 1.0 for perfect predictions"

    # Case 2: All zeros prediction
    y_pred_zeros = np.array([0.0, 0.0, 0.0, 0.0])
    score_zeros = probabilistic_f1(y_true, y_pred_zeros)
    print(f"pF1 (All Zeros): {score_zeros:.4f}")
    assert score_zeros == 0.0, "pF1 should be 0.0 for all zero predictions"

    # Case 3: Mixed probabilities
    y_pred_mixed = np.array([0.8, 0.2, 0.6, 0.1])
    # pTP = 1*0.8 + 0*0.2 + 1*0.6 + 0*0.1 = 1.4
    # sum_pred = 0.8 + 0.2 + 0.6 + 0.1 = 1.7
    # total_pos = 2
    # pPrec = 1.4 / 1.7 = 0.8235
    # pRec = 1.4 / 2.0 = 0.7
    # pF1 = 2 * (0.8235 * 0.7) / (0.8235 + 0.7) = 1.1529 / 1.5235 = 0.7567
    score_mixed = probabilistic_f1(y_true, y_pred_mixed)
    print(f"pF1 (Mixed): {score_mixed:.4f}")

    # Allow small float error
    assert 0.75 < score_mixed < 0.76, f"pF1 calculation incorrect. Got {score_mixed}"

    print("Metric verification passed.")
    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
