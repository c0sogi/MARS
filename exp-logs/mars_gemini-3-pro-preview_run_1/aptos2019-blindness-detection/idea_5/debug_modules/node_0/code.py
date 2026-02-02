import sys
import os
import torch
import pandas as pd
import numpy as np

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, compute_score
from library.data import get_dataloaders
from library.model import RetinopathyModel
from library.engine import train_one_epoch, validate, generate_submission


def run_demo():
    print("=== Starting Diabetic Retinopathy Task Demo ===")

    # 1. Configuration Setup
    # Modify Config for a fast, verifiable demonstration
    Config.debug = True  # Use small subset of data (32 samples)
    Config.epochs = 1  # Run only 1 epoch
    Config.batch_size = 4
    Config.num_workers = 0  # Avoid multiprocessing overhead for small demo
    Config.working_dir = "./working/demo_run"
    Config.model_save_path = os.path.join(Config.working_dir, "best_model.pth")
    Config.submission_path = os.path.join(Config.working_dir, "submission.csv")

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # Set seeds
    seed_everything(Config.seed)
    print(f"Configuration: Debug={Config.debug}, Batch Size={Config.batch_size}")

    # 2. Data Loading Verification
    print("\n--- Verifying Data Loading ---")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.batch_size, num_workers=Config.num_workers, debug=Config.debug
    )

    # Fetch one batch
    images, targets = next(iter(train_loader))

    print(f"Image Batch Shape: {images.shape}")
    print(f"Target Batch Shape: {targets.shape}")

    # Assertions for Data
    assert images.shape == (
        Config.batch_size,
        3,
        Config.image_size,
        Config.image_size,
    ), f"Expected image shape {(Config.batch_size, 3, Config.image_size, Config.image_size)}, got {images.shape}"
    assert targets.shape == (
        Config.batch_size,
        Config.num_outputs,
    ), f"Expected target shape {(Config.batch_size, Config.num_outputs)}, got {targets.shape}"

    # Verify Ordinal Encoding consistency
    # For ordinal regression, if index i is 1, index i-1 must be 1.
    # Example: [1, 1, 0, 0] is valid (Label 2). [0, 1, 0, 0] is invalid.
    for i in range(targets.shape[0]):
        t = targets[i]
        # Sort descending should match original if the sequence is valid (1s followed by 0s)
        t_sorted, _ = torch.sort(t, descending=True)
        assert torch.equal(
            t, t_sorted
        ), f"Invalid ordinal target encoding found: {t}. 1s must precede 0s."
    print("Data loading and ordinal encoding verified.")

    # 3. Model & Training Verification
    print("\n--- Verifying Model and Training Loop ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Model (pretrained=False for speed)
    model = RetinopathyModel(pretrained=False)
    model.to(device)

    # Forward Pass Check
    images = images.to(device)
    logits = model(images)
    assert logits.shape == (
        Config.batch_size,
        Config.num_outputs,
    ), f"Model output shape mismatch. Expected {(Config.batch_size, Config.num_outputs)}, got {logits.shape}"

    # Define Training Components
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)

    # Run Training Step
    print("Running train_one_epoch...")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss returned NaN."

    # Run Validation Step
    print("Running validate...")
    val_loss, val_qwk = validate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, QWK: {val_qwk:.4f}")
    assert not np.isnan(val_loss), "Validation loss returned NaN."
    assert (
        -1.0 <= val_qwk <= 1.0
    ), f"QWK score {val_qwk} is out of theoretical range [-1, 1]."

    # 4. Submission Verification
    print("\n--- Verifying Submission Generation ---")
    # Save the current model state so generate_submission can load it
    torch.save(model.state_dict(), Config.model_save_path)
    assert os.path.exists(Config.model_save_path), "Failed to save model checkpoint."

    # Run the submission generation function from engine.py
    generate_submission()

    # Verify output file
    assert os.path.exists(Config.submission_path), "Submission file was not created."

    df_sub = pd.read_csv(Config.submission_path)
    print(f"Submission file loaded. Rows: {len(df_sub)}")
    print(df_sub.head())

    # Verify submission content
    assert (
        "id_code" in df_sub.columns and "diagnosis" in df_sub.columns
    ), "Submission file missing required columns."
    assert len(df_sub) > 0, "Submission file is empty."
    # Since debug=True, we expect a small number of rows (up to 32)
    assert (
        df_sub["diagnosis"].between(0, 4).all()
    ), "Predictions contain values outside [0, 4]."

    # 5. Metric Utility Verification
    print("\n--- Verifying Metric Utility ---")
    # Perfect agreement
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred_perfect = np.array([0, 1, 2, 3, 4])
    score_perfect = compute_score(y_true, y_pred_perfect)
    assert np.isclose(
        score_perfect, 1.0
    ), f"Metric failed perfect agreement check. Got {score_perfect}"

    # Random/Bad agreement
    y_pred_bad = np.array([4, 3, 0, 1, 0])
    score_bad = compute_score(y_true, y_pred_bad)
    print(f"Perfect Score: {score_perfect}, Bad Score: {score_bad}")
    assert score_bad < 1.0, "Metric failed bad agreement check."

    print("\n=== All Tests Passed Successfully ===")


if __name__ == "__main__":
    run_demo()
