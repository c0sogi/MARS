import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import DeepSupervisionUNet
from library.train import train_model
from library.inference import predict_with_ensemble, create_submission_file, TTAHandler


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Set a separate working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Limit data and training parameters for speed
    Config.MAX_SAMPLES = 20  # Use only 20 images for train/val/test
    Config.NUM_EPOCHS = 2  # Train for only 2 epochs
    Config.NUM_MODELS = 1  # Train only 1 model (instead of ensemble of 5)
    Config.BATCH_SIZE = 4  # Small batch size
    Config.PATCH_SIZE = 128  # Slightly smaller patch size

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed
    seed_everything(Config.SEED)
    print("Configuration updated.")

    # -------------------------------------------------------------------------
    # 2. Verify Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Force reload from source to ensure we use the limited MAX_SAMPLES
    # We pass load_cached_data=False to ignore any pre-existing full-dataset cache
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Check Training Batch (Should be cropped to PATCH_SIZE)
    noisy_batch, clean_batch = next(iter(train_loader))
    print(
        f"Training Batch Shape - Noisy: {noisy_batch.shape}, Clean: {clean_batch.shape}"
    )

    assert noisy_batch.shape == (
        Config.BATCH_SIZE,
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), "Training batch dimensions mismatch (expected patch size)."
    assert clean_batch.shape == (
        Config.BATCH_SIZE,
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), "Training target dimensions mismatch."

    # Check Validation Batch (Should be full size, Batch size 1)
    val_noisy, val_clean = next(iter(val_loader))
    print(f"Validation Batch Shape - Noisy: {val_noisy.shape}")
    assert val_noisy.shape[0] == 1, "Validation batch size must be 1."
    # Note: Spatial dimensions vary per image in validation, so we just check rank
    assert val_noisy.ndim == 4, "Validation tensor should be 4D (B, C, H, W)."

    print("Data Loading verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = DeepSupervisionUNet().to(device)

    # Create dummy input
    dummy_input = torch.randn(2, 1, 128, 128).to(device)

    # Test Training Mode (Deep Supervision enabled -> List of outputs)
    model.train()
    outputs_train = model(dummy_input)
    print(f"Model Output Type (Train): {type(outputs_train)}")

    if Config.DEEP_SUPERVISION:
        assert isinstance(
            outputs_train, list
        ), "Model should return a list in training mode with Deep Supervision."
        print(f"Number of Deep Supervision outputs: {len(outputs_train)}")
        # Check shape of final output
        assert (
            outputs_train[0].shape == dummy_input.shape
        ), "Final output shape mismatch."
    else:
        assert torch.is_tensor(
            outputs_train
        ), "Model should return tensor if Deep Supervision is disabled."

    # Test Eval Mode (Single output)
    model.eval()
    with torch.no_grad():
        output_eval = model(dummy_input)
    print(f"Model Output Type (Eval): {type(output_eval)}")
    assert torch.is_tensor(
        output_eval
    ), "Model should return a single tensor in eval mode."
    assert output_eval.shape == dummy_input.shape, "Eval output shape mismatch."

    print("Model architecture verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Run Training Loop
    # -------------------------------------------------------------------------
    print("\n[4] Running Training Loop (Demo)...")

    # Train model index 0
    best_rmse = train_model(model_index=0, load_cached_data=True)

    print(f"Training completed. Best RMSE: {best_rmse}")
    model_path = os.path.join(Config.WORKING_DIR, "model_0.pth")
    assert os.path.exists(model_path), "Model checkpoint file was not created."

    print("Training loop verified successfully.")

    # -------------------------------------------------------------------------
    # 5. Verify TTA (Test-Time Augmentation) Logic
    # -------------------------------------------------------------------------
    print("\n[5] Verifying TTA Logic...")

    tta = TTAHandler()

    # Create a synthetic image with a distinct pattern to verify orientation
    # Gradient from top-left to bottom-right
    h, w = 100, 100
    img_np = np.zeros((1, 1, h, w), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            img_np[0, 0, r, c] = (r + c) / (h + w)

    img_tensor = torch.from_numpy(img_np)

    # Apply transforms
    augmented = tta.apply_transforms(img_tensor)  # Shape (8, 1, H, W)
    assert augmented.shape == (8, 1, h, w), "TTA augmented batch shape mismatch."

    # Reverse transforms
    # If we reverse the augmented images immediately, we should get the original image back 8 times
    restored = tta.reverse_transforms(augmented)

    # Check consistency
    # We allow small floating point errors
    diff = torch.abs(restored - img_tensor).max().item()
    print(f"Max difference after TTA round-trip: {diff:.6f}")
    assert diff < 1e-5, "TTA round-trip failed to restore original image."

    print("TTA logic verified successfully.")

    # -------------------------------------------------------------------------
    # 6. Run Inference
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference...")

    # Predict using the trained model (and TTA)
    predictions = predict_with_ensemble(load_cached_data=True)

    print(f"Number of predictions generated: {len(predictions)}")
    assert len(predictions) > 0, "No predictions were generated."

    # Check one prediction
    sample_id = list(predictions.keys())[0]
    sample_pred = predictions[sample_id]
    print(f"Sample prediction shape for ID {sample_id}: {sample_pred.shape}")

    assert isinstance(sample_pred, np.ndarray), "Prediction should be a numpy array."
    assert sample_pred.ndim == 2, "Prediction should be 2D (H, W)."

    print("Inference verified successfully.")

    # -------------------------------------------------------------------------
    # 7. Generate Submission File
    # -------------------------------------------------------------------------
    print("\n[7] Generating Submission File...")

    create_submission_file(predictions)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    # Verify file content format
    df = pd.read_csv(Config.SUBMISSION_PATH, nrows=5)
    print("First 5 rows of submission:")
    print(df)

    assert list(df.columns) == ["id", "value"], "Submission columns mismatch."
    assert "_" in str(df.iloc[0]["id"]), "ID format mismatch (expected underscores)."

    print("Submission generation verified successfully.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
