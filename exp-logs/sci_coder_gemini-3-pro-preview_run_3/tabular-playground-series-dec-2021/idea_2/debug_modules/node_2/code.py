import os
import torch
import numpy as np
import pandas as pd
import shutil
import warnings

# Import library modules
from library.utils import (
    seed_everything,
    inverse_transform_target,
    compute_metrics,
    TARGET_CLASSES,
)
from library.data_loader import get_dataloaders, ForestDataset
from library.model import ResNetMLP, train_one_epoch, validate
from library.train import run_training
from library.inference import generate_submission

# Configuration for the demo
DEBUG_LIMIT = 2000
BATCH_SIZE = 128
EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # 1. Setup
    print("=== Step 1: Setup & Reproducibility ===")
    seed_everything(42)
    warnings.filterwarnings("ignore")
    print("Random seed set to 42.")

    # 2. Verify Utils
    print("\n=== Step 2: Verifying Utils ===")

    # Test inverse_transform_target
    # We expect indices 0, 1, 2 to map to the first three classes in TARGET_CLASSES
    dummy_preds = np.array([0, 1, 2])
    mapped_preds = inverse_transform_target(dummy_preds)
    expected_preds = np.array([TARGET_CLASSES[0], TARGET_CLASSES[1], TARGET_CLASSES[2]])

    assert np.array_equal(
        mapped_preds, expected_preds
    ), f"Inverse transform failed. Got {mapped_preds}, expected {expected_preds}"
    print("inverse_transform_target: Verified.")

    # Test compute_metrics
    y_true = np.array([1, 2, 3, 1])
    y_pred_correct = np.array([1, 2, 3, 1])
    y_pred_wrong = np.array([1, 2, 3, 2])

    acc_perfect = compute_metrics(y_true, y_pred_correct)
    acc_imperfect = compute_metrics(y_true, y_pred_wrong)

    assert acc_perfect == 1.0, "Metric computation failed for perfect predictions."
    assert acc_imperfect == 0.75, "Metric computation failed for imperfect predictions."
    print("compute_metrics: Verified.")

    # 3. Verify Data Loader
    print("\n=== Step 3: Verifying Data Loader ===")
    # Use debug_limit to speed up loading
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=2,
        load_cached_data=True,  # Will process and cache if not present
        debug_limit=DEBUG_LIMIT,
    )

    # Check loader lengths
    assert (
        len(train_loader.dataset) == DEBUG_LIMIT
    ), f"Train dataset size mismatch. Expected {DEBUG_LIMIT}, got {len(train_loader.dataset)}"

    # Fetch a batch
    X_batch, y_batch = next(iter(train_loader))

    # Check shapes
    # Feature dim should be 54 (10 numerical + 44 binary)
    assert (
        X_batch.shape[1] == 54
    ), f"Feature dimension mismatch. Expected 54, got {X_batch.shape[1]}"
    assert y_batch.shape[0] == X_batch.shape[0], "Batch size mismatch between X and y."

    print(f"Data Loaders created successfully. Batch shape: {X_batch.shape}")

    # 4. Verify Model Architecture
    print("\n=== Step 4: Verifying Model Architecture ===")
    input_dim = 54
    num_classes = 6  # Mapped classes

    model = ResNetMLP(
        input_dim=input_dim, num_classes=num_classes, num_blocks=2, hidden_dim=128
    ).to(DEVICE)

    # Forward pass check
    X_batch = X_batch.to(DEVICE)
    with torch.no_grad():
        outputs = model(X_batch)

    assert outputs.shape == (
        X_batch.shape[0],
        num_classes,
    ), f"Model output shape mismatch. Expected {(X_batch.shape[0], num_classes)}, got {outputs.shape}"

    print("Model forward pass successful.")

    # 5. Verify Training Pipeline
    print("\n=== Step 5: Verifying Training Pipeline (run_training) ===")
    # This function handles the full loop: train, val, early stopping, and prediction
    # We run for 1 epoch with the debug dataset

    try:
        run_training(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=1e-3,
            patience=1,
            debug_limit=DEBUG_LIMIT,
            num_workers=2,
        )
        print("run_training executed successfully.")
    except Exception as e:
        raise AssertionError(f"run_training failed with error: {e}")

    # Check if submission file was created
    submission_path = "./submission/submission.csv"
    assert os.path.exists(
        submission_path
    ), "Submission file was not created by run_training."

    df_sub = pd.read_csv(submission_path)
    assert (
        "Id" in df_sub.columns and "Cover_Type" in df_sub.columns
    ), "Submission file missing required columns."
    assert (
        len(df_sub) == DEBUG_LIMIT
    ), f"Submission length mismatch. Expected {DEBUG_LIMIT} (debug limit), got {len(df_sub)}"

    print("Submission file verified.")

    # 6. Verify Inference Module
    print("\n=== Step 6: Verifying Inference Module ===")

    # Save the current model state to simulate a trained model file
    model_path = "./working/demo_model.pth"
    torch.save(model.state_dict(), model_path)

    inference_output = "./submission/inference_submission.csv"

    # Run inference using the saved model
    generate_submission(
        model_path=model_path,
        output_path=inference_output,
        batch_size=BATCH_SIZE,
        num_workers=2,
        device=DEVICE,
        num_blocks=2,
        hidden_dim=128,
    )

    assert os.path.exists(inference_output), "Inference submission file not found."
    df_inf = pd.read_csv(inference_output)
    assert len(df_inf) > 0, "Inference submission is empty."

    print("Inference module verified.")

    print("\n=== All Demonstrations Passed Successfully ===")


if __name__ == "__main__":
    main()
