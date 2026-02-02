import os
import torch
import pandas as pd
import numpy as np
import shutil

from library.utils import seed_everything, quadratic_weighted_kappa
from library.dataset import get_dataloaders
from library.model import RetinopathyModel
from library.train import train
from library.predict import inference_fn


def run_demo():
    print("=== Starting Diabetic Retinopathy Task Demo ===")

    # 1. Setup and Utility Verification
    print("\n[1/5] Verifying Utilities...")
    seed_everything(42)

    # Verify Quadratic Weighted Kappa
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 3, 4]
    score = quadratic_weighted_kappa(y_true, y_pred)
    assert score == 1.0, f"Expected QWK of 1.0 for perfect match, got {score}"

    y_pred_bad = [4, 3, 2, 1, 0]
    score_bad = quadratic_weighted_kappa(y_true, y_pred_bad)
    assert score_bad < 0, f"Expected negative QWK for inverse match, got {score_bad}"
    print("Utilities verified.")

    # 2. Data Pipeline Verification
    # This step triggers the processing (crop/resize) of images and caches them.
    # Subsequent calls in train/predict will leverage the cache.
    print("\n[2/5] Verifying Data Pipeline (Processing & Loading)...")
    batch_size = 4
    image_size = 512

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, image_size=image_size, load_cached_data=True
    )

    # Fetch a single batch to verify shapes
    images, labels = next(iter(train_loader))

    # Expected shape: (Batch, Channels, Height, Width)
    assert images.shape == (
        batch_size,
        3,
        image_size,
        image_size,
    ), f"Image batch shape mismatch. Expected {(batch_size, 3, image_size, image_size)}, got {images.shape}"

    # Expected labels: (Batch,)
    assert labels.shape == (
        batch_size,
    ), f"Label batch shape mismatch. Expected {(batch_size,)}, got {labels.shape}"

    print(f"Data loaded successfully. Batch Shape: {images.shape}")

    # 3. Model Architecture Verification
    print("\n[3/5] Verifying Model Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model (pretrained=False for speed in this demo)
    model = RetinopathyModel(pretrained=False)
    model.to(device)

    # Forward pass check
    input_tensor = images.to(device)
    with torch.no_grad():
        output = model(input_tensor)

    # Expected output: (Batch, 1) for regression
    assert output.shape == (
        batch_size,
        1,
    ), f"Model output shape mismatch. Expected {(batch_size, 1)}, got {output.shape}"
    print("Model forward pass verified.")

    # 4. Training Loop Integration
    print("\n[4/5] Running Training Loop (Debug Subset)...")
    working_dir = "./working/demo_run"
    submission_path = os.path.join(working_dir, "submission.csv")

    # Clean up previous demo run if exists
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)

    # Run training on a tiny subset of data for 1 epoch to verify the loop
    train(
        epochs=1,
        batch_size=4,
        learning_rate=1e-4,
        patience=1,
        debug_subset_size=16,  # Train on only 16 samples for speed
        save_dir=working_dir,
        submission_path=submission_path,
    )

    # Verify model checkpoint was saved
    best_model_path = os.path.join(working_dir, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not saved."
    print("Training loop completed and model saved.")

    # 5. Inference Integration
    print("\n[5/5] Running Inference...")
    inference_output = os.path.join(working_dir, "final_submission.csv")

    inference_fn(
        model_path=best_model_path,
        output_path=inference_output,
        batch_size=4,
        image_size=image_size,
        device=device,
    )

    # Verify submission file
    assert os.path.exists(
        inference_output
    ), "Inference did not generate a submission file."

    df_sub = pd.read_csv(inference_output)
    assert "id_code" in df_sub.columns, "Submission missing 'id_code' column."
    assert "diagnosis" in df_sub.columns, "Submission missing 'diagnosis' column."
    assert len(df_sub) > 0, "Submission file is empty."

    # Verify values are integers (as required by QWK and submission format)
    assert pd.api.types.is_integer_dtype(
        df_sub["diagnosis"]
    ), "Diagnosis column should be integers."

    print(f"Inference successful. Submission saved to {inference_output}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
